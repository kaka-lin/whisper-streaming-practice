import torch

# This is copied from silero-vad's vad_utils.py:
# https://github.com/snakers4/silero-vad/blob/f6b1294cb27590fb2452899df98fb234dfef1134/utils_vad.py#L340
# (except changed defaults)

# Their licence is MIT, same as ours: https://github.com/snakers4/silero-vad/blob/f6b1294cb27590fb2452899df98fb234dfef1134/LICENSE

class VADIterator:
    """ VADIterator
    這個 iterator 是為了要解決 silero-vad 原生模型的問題。
    silero-vad 原生模型只能處理 512 samples 長度的音訊，這樣才能正確地判斷出音訊裡面有沒有說話的聲音。
    但是如果音訊長度不是 512 samples，silero-vad 就會報錯。
    這個 iterator 的功能就是把音訊切成一段一段的，然後傳進去給 silero-vad 判斷。

    Parameters
    ----------
    model: preloaded .jit silero VAD model

    threshold: float (default - 0.5)
        Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
        It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

    sampling_rate: int (default - 16000)
        Currently silero VAD models support 8000 and 16000 sample rates

    min_silence_duration_ms: int (default - 500 milliseconds)
        In the end of each speech chunk wait for min_silence_duration_ms before separating it

    speech_pad_ms: int (default - 100 milliseconds)
        Final speech chunks are padded by speech_pad_ms each side.
        在 speech 開頭和結尾稍微往外多留一點 margin。
    """
    def __init__(self,
                 model,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 500,  # makes sense on one recording that I checked
                 speech_pad_ms: int = 100             # same
                 ):

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        """Reset internal states of the iterator."""
        self.model.reset_states()
        self.triggered = False   # 現在是不是正在 inside speech
        self.temp_end = 0        # 暫時紀錄靜音的開始點
        self.current_sample = 0  # 現在跑到第幾個 sample

    def __call__(self, x, return_seconds=False):
        """ 傳進一段 audio，然後判斷這段 audio 是不是有 speech

        Parameters
            x: torch.Tensor
               audio chunk (see examples in repo)

            return_seconds: bool (default - False)
                whether return timestamps in seconds (default - samples)
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        # 高於門檻，且之前是靜音 -> 標記開始說話
        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = self.current_sample - self.speech_pad_samples
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, 1)}

        # 低於門檻，且正在說話 -> 標記結束說話
        # self.threshold - 0.15: 讓結束偵測更敏感一點（提早收尾）
        # 這樣可以避免說話結束後，還有一小段音量很小的聲音
        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample

            # 如果靜音的時間小於 min_silence_samples，短暫靜音，不會立刻結束
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, 1)}

        return None

#######################
# because Silero now requires exactly 512-sized audio chunks

import numpy as np


class FixedVADIterator(VADIterator):
    ''' FixedVADIterator
    這是 一個修正過的 VADIterator，讓它可以處理任意長度的音訊。

    It fixes VADIterator by allowing to process any audio length, not only exactly 512 frames at once.
    If audio to be processed at once is long and multiple voiced segments detected,
    then __call__ returns the start of the first segment, and end (or middle, which means no end) of the last segment.
    '''

    def reset_states(self):
        super().reset_states()
        self.buffer = np.array([], dtype=np.float32)

    def __call__(self, x, return_seconds=False):
        # Append the new audio to the buffer
        self.buffer = np.append(self.buffer, x)

        # If the buffer is larger than 512, process it
        ret = None
        while len(self.buffer) >= 512:
            r = super().__call__(self.buffer[:512], return_seconds=return_seconds)
            self.buffer = self.buffer[512:]
            if ret is None:
                ret = r
            elif r is not None:
                if 'end' in r:
                    ret['end'] = r['end']  # the latter end

                # 如果有新的 start 發生在上一個 end 之後，會自動 meerge 成同一個段落
                if 'start' in r and 'end' in ret:  # there is an earlier start.
                    # Remove end, merging this segment with the previous one.
                    del ret['end']
        return ret if ret != {} else None


if __name__ == "__main__":
    # test/demonstrate the need for FixedVADIterator:

    import torch
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad'
    )
    vac = FixedVADIterator(model)
    # vac = VADIterator(model)  # the second case crashes with this

    # this works: for both
    audio_buffer = np.array([0]*(512), dtype=np.float32)
    vac(audio_buffer)

    # this crashes on the non FixedVADIterator with
    # ops.prim.RaiseException("Input audio chunk is too short", "builtins.ValueError")
    audio_buffer = np.array([0]*(512-1), dtype=np.float32)
    vac(audio_buffer)
