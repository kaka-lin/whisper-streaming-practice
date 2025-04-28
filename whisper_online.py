#!/usr/bin/env python3
import sys
import numpy as np
import librosa
from functools import lru_cache
import time
import logging

import io
import soundfile as sf
import math

logger = logging.getLogger(__name__)

@lru_cache(10**6)
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]


# Whisper backend

class ASRBase:

    sep = " "   # join transcribe words with this character (" " for whisper_timestamped,
                # "" for faster-whisper because it emits the spaces when neeeded)

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.logfile = logfile

        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan

        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize, cache_dir):
        raise NotImplemented("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplemented("must be implemented in the child class")

    def use_vad(self):
        raise NotImplemented("must be implemented in the child class")


class WhisperTimestampedASR(ASRBase):
    """Uses whisper_timestamped library as the backend. Initially, we tested the code on this backend. It worked, but slower than faster-whisper.
    On the other hand, the installation for GPU could be easier.
    """

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        import whisper
        import whisper_timestamped
        from whisper_timestamped import transcribe_timestamped
        self.transcribe_timestamped = transcribe_timestamped
        if model_dir is not None:
            logger.debug("ignoring model_dir, not implemented")
        return whisper.load_model(modelsize, download_root=cache_dir)

    def transcribe(self, audio, init_prompt=""):
        result = self.transcribe_timestamped(self.model,
                audio, language=self.original_language,
                initial_prompt=init_prompt, verbose=None,
                condition_on_previous_text=True, **self.transcribe_kargs)
        return result

    def ts_words(self,r):
        # return: transcribe result object to [(beg,end,"word1"), ...]
        o = []
        for s in r["segments"]:
            for w in s["words"]:
                t = (w["start"],w["end"],w["text"])
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s["end"] for s in res["segments"]]

    def use_vad(self):
        self.transcribe_kargs["vad"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class FasterWhisperASR(ASRBase):
    """Uses faster-whisper library as the backend.

    Works much faster, appx 4-times (in offline mode).
    For GPU, it requires installation with a specific CUDNN version.
    """
    sep = ""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel
        # logging.getLogger("faster_whisper").setLevel(logger.level)
        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")

        # this worked fast and reliably on NVIDIA L40
        model = WhisperModel(model_size_or_path, device="cuda", compute_type="float16", download_root=cache_dir)

        # or run on GPU with INT8
        # tested: the transcripts were different, probably worse than with FP16, and it was slightly (appx 20%) slower
        # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

        # or run on CPU with INT8
        # tested: works, but slow, appx 10-times than cuda FP16
        # model = WhisperModel(modelsize, device="cpu", compute_type="int8") #, download_root="faster-disk-cache-dir/")
        return model

    def transcribe(self, audio, init_prompt=""):

        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs
        )

        return list(segments)

    def ts_words(self, segments):
        tokens = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                # not stripping the spaces -- should not be merged with them!
                tokens.append((word.start, word.end, word.word))
        return tokens

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class MLXWhisper(ASRBase):
    """
    Uses MLX Whisper library as the backend, optimized for Apple Silicon.
    Models available: https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc
    Significantly faster than faster-whisper (without CUDA) on Apple M1.
    """
    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        """
        Loads the MLX-compatible Whisper model.

        Args:
            modelsize (str, optional): The size or name of the Whisper model to load.
                If provided, it will be translated to an MLX-compatible model path using the `translate_model_name` method.
                Example: "large-v3-turbo" -> "mlx-community/whisper-large-v3-turbo".
            cache_dir (str, optional): Path to the directory for caching models.
                **Note**: This is not supported by MLX Whisper and will be ignored.
            model_dir (str, optional): Direct path to a custom model directory.
                If specified, it overrides the `modelsize` parameter.
        """
        from mlx_whisper.transcribe import ModelHolder, transcribe
        import mlx.core as mx # Is installed with mlx-whisper

        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize parameter is not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = self.translate_model_name(modelsize)
            logger.debug(f"Loading whisper model {modelsize}. You use mlx whisper, so {model_size_or_path} will be used.")

        self.model_size_or_path = model_size_or_path

        # Note: ModelHolder.get_model loads the model into a static class variable,
        # making it a global resource. This means:
        # - Only one model can be loaded at a time; switching models requires reloading.
        # - This approach may not be suitable for scenarios requiring multiple models simultaneously,
        #   such as using whisper-streaming as a module with varying model sizes.
        dtype = mx.float16 # Default to mx.float16. In mlx_whisper.transcribe: dtype = mx.float16 if decode_options.get("fp16", True) else mx.float32
        ModelHolder.get_model(model_size_or_path, dtype) #Model is preloaded to avoid reloading during transcription

        return transcribe

    def translate_model_name(self, model_name):
        """
        Translates a given model name to its corresponding MLX-compatible model path.

        Args:
            model_name (str): The name of the model to translate.

        Returns:
            str: The MLX-compatible model path.
        """
        # Dictionary mapping model names to MLX-compatible paths
        model_mapping = {
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx"
        }

        # Retrieve the corresponding MLX model path
        mlx_model_path = model_mapping.get(model_name)

        if mlx_model_path:
            return mlx_model_path
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized or not supported.")

    def transcribe(self, audio, init_prompt=""):
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
            **self.transcribe_kargs
        )
        return segments.get("segments", [])

    def ts_words(self, segments):
        """
        Extract timestamped words from transcription segments and skips words with high no-speech probability.
        """
        return [
            (word["start"], word["end"], word["word"])
            for segment in segments
            for word in segment.get("words", [])
            if segment.get("no_speech_prob", 0) <= 0.9
        ]

    def segments_end_ts(self, res):
        return [s['end'] for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class OpenaiApiASR(ASRBase):
    """Uses OpenAI's Whisper API for audio transcription."""

    def __init__(self, lan=None, temperature=0, logfile=sys.stderr):
        self.logfile = logfile

        self.modelname = "whisper-1"
        self.original_language = None if lan == "auto" else lan # ISO-639-1 language code
        self.response_format = "verbose_json"
        self.temperature = temperature

        self.load_model()

        self.use_vad_opt = False

        # reset the task in set_translate_task
        self.task = "transcribe"

    def load_model(self, *args, **kwargs):
        from openai import OpenAI
        self.client = OpenAI()

        self.transcribed_seconds = 0  # for logging how many seconds were processed by API, to know the cost


    def ts_words(self, segments):
        no_speech_segments = []
        if self.use_vad_opt:
            for segment in segments.segments:
                # TODO: threshold can be set from outside
                if segment["no_speech_prob"] > 0.8:
                    no_speech_segments.append((segment.get("start"), segment.get("end")))

        o = []
        for word in segments.words:
            start = word.start
            end = word.end
            if any(s[0] <= start <= s[1] for s in no_speech_segments):
                # print("Skipping word", word.get("word"), "because it's in a no-speech segment")
                continue
            o.append((start, end, word.word))
        return o


    def segments_end_ts(self, res):
        return [s.end for s in res.words]

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        # Write the audio data to a buffer
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format='WAV', subtype='PCM_16')
        buffer.seek(0)  # Reset buffer's position to the beginning

        self.transcribed_seconds += math.ceil(len(audio_data)/16000)  # it rounds up to the whole seconds

        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"]
        }
        if self.task != "translate" and self.original_language:
            params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt

        if self.task == "translate":
            proc = self.client.audio.translations
        else:
            proc = self.client.audio.transcriptions

        # Process transcription/translation
        transcript = proc.create(**params)
        logger.debug(f"OpenAI API processed accumulated {self.transcribed_seconds} seconds")

        return transcript

    def use_vad(self):
        self.use_vad_opt = True

    def set_translate_task(self):
        self.task = "translate"


class HypothesisBuffer:
    """
    HypothesisBuffer is a buffer management class for streaming ASR results.
    It helps stabilize output by comparing hypotheses between frames and removing
    duplicate or repeated words due to model latency.

    ✅ Responsibilities:
    - Track committed (stable) segments of transcription
    - Remove repeated n-grams when model re-emits past outputs
    - Buffer incoming hypothesis and flush stable words
    - Used in real-time captioning or streaming ASR pipelines
    """

    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = []  # Commited words: (start_time, end_time, word)
        self.buffer = []              # Last buffers state to compare with current (not yet commited)
        self.new = []                 # New hypothesis in current frame

        self.last_commited_time = 0   # Timestamp of last committed word
        self.last_commited_word = None

        self.logfile = logfile

    def insert(self, new, offset):
        """
        Compares a `new` hypothesis with previously committed content `self.commited_in_buffer` and removes redundancy.

        - Adjusts timestamps in the incoming `new` word list using the provided `offset`.
        - Filters out words that occur too close to or before the `last_commited_time`.
        - Compares the tail of the committed buffer (`commited_in_buffer`) with the head of the new hypothesis.
          If a duplicate n-gram (1 to 5 words) is found, it is removed from the new hypothesis.
        - The remaining non-duplicate words are stored in `self.new` for further validation and commitment.
        """

        # Adjust timestamps by the offset
        new = [(a+offset, b+offset, t) for a, b, t in new]

        # Keep only new words that start after last committed time
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        # Try to detect if the beginning of the `new` hypothesis is repeated n-grams from the commited buffer
        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    # Compare up to 5-gram from end of committed buffer and start of new
                    for i in range(1, min(min(cn, nn), 5) + 1):  # 5 is the maximum
                        # Compare the last i words in the committed buffer with the first i words in the new hypothesis
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1, i+1)][::-1])
                        tail = " ".join(self.new[j-1][2] for j in range(1, i+1))
                        if c == tail:
                            # Remove duplicate n-gram from the beginning of new
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = " ".join(words)
                            logger.debug(f"removing last {i} words: {words_msg}")
                            break

    def flush(self):
        """
        Commits the overlapping (common) words between the last `buffer` and the current `new` hypothesis.

        - Compares `self.new` and `self.buffer` word by word.
        - As long as the words (by token) match in order, they are considered committed and moved into the result.
        - Updates `last_commited_word` and `last_commited_time` based on the committed chunk.
        - The remaining `self.new` becomes the new `self.buffer` (for next insert).
        - Clears `self.new` for the next cycle.

        Returns:
            A list of committed (timestamped) words tuples: [(start_time, end_time, word), ...]
        """

        # Commit longest matching prefix between previous buffer and new hypothesis
        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        """
        Remove committed words from the buffer that have already ended before the given time.

        Args:
            time (float): Timestamp threshold (in seconds). All words with end_time <= time will be removed
        """
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        """
        Returns:
            The current buffer state, representing uncommitted (but most recent) recognized words.
        """
        return self.buffer


class OnlineASRProcessor:
    """
    A real-time streaming Automatic Speech Recognition (ASR) processor.

    This class handles streaming audio input and performs real-time ASR using a backend model (e.g., Whisper).
    It manages audio buffering, prepares prompt/context for continuity, and processes recognition results
    to produce stable, non-redundant transcription output.

    Key Features:
        - Maintains synchronized audio and transcript buffers.
        - Uses `HypothesisBuffer` to track and deduplicate hypotheses across iterations.
        - Supports buffer trimming based on complete sentences (with a tokenizer) or model-predicted segments.
        - Prepares contextual prompts to improve recognition continuity.
        - Flushes and finalizes incomplete text on stream completion.

    Args:
        asr (WhisperASR): ASR model object with methods `transcribe()` and `ts_words()`.
        tokenizer (object, optional): Sentence tokenizer (must support `.split()`); only required for sentence-based trimming.
        buffer_trimming (tuple): A pair like ("segment", 15) or ("sentence", 30), defining trimming strategy and threshold (in seconds).
        logfile (file-like): Log output stream (e.g., sys.stderr).
    """

    SAMPLING_RATE = 16000

    def __init__(self, asr, tokenizer=None, buffer_trimming=("segment", 15), logfile=sys.stderr):
        """
        Args:
            asr: WhisperASR object
            tokenizer: Sentence tokenizer object (e.g. MosesTokenizer-like) for the target language with `.split(text)`.
                       It can be None, if "segment" buffer trimming option is used, then tokenizer is not used at all.
            buffer_trimming: Tuple ("segment" or "sentence", seconds) and seconds is a number, like: ("segment", 15).
                             Buffer is trimmed if it is longer than "seconds" threshold.
                             Default is the most recommended option.
            logfile: where to store the log.
        """
        self.asr = asr
        self.tokenizer = tokenizer  # Tokenizer for sentence segmentation (optional)
        self.logfile = logfile

        self.init()

        # Unpack trimming strategy: mode and threshold (in seconds)
        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

    def init(self, offset=None):
        """Run this when starting or restarting processing."""
        # Initialize the audio buffer to an empty float32 array
        self.audio_buffer = np.array([], dtype=np.float32)

        # Create a new hypothesis buffer to manage interim and committed transcriptions
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)

        # Reset time offset for the buffer
        self.buffer_time_offset = 0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset

        # Clear all committed transcript segments
        self.commited = []

    def insert_audio_chunk(self, audio):
        """Inserts a chunk of audio data into the buffer."""
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """
        Returns a tuple: (prompt, context)

        - prompt: Up to 200 characters of previously committed transcript that is no longer in the current audio buffer.
                  This is used as a prompt to help the model understand the current context.
        - context: The committed text that is still within the audio buffer.  It is transcribed again and skipped
                   This part is not included in the prompt and is only for debugging/logging.

        Example:
            commited = [(0.0, 0.5, "hello"), (0.5, 1.0, "my"), (1.0, 1.5, "name"), (1.5, 2.0, "is"), (2.0, 2.5, "John")]
            buffer_time_offset = 1.2
            prompt = "hello my name"
            context = "is John"
        """
        # Find the last committed word that is before the current buffer time offset
        # This is used to determine the prompt context.
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k-1][1] > self.buffer_time_offset:
            k -= 1

        # Extract committed segments up to k
        committed = self.commited[:k]
        # Extract text only
        committed_texts = [t for _, _, t in committed]

        # Build prompt by taking from the end until reaching MAX_PROMPT_CHARS
        MAX_PROMPT_CHARS = 200
        prompt_pieces = []
        length = 0
        while committed_texts and length < MAX_PROMPT_CHARS:  # 200 characters prompt size
            piece = committed_texts.pop() # take the last piece
            prompt_pieces.append(piece)
            length += len(piece) + len(self.asr.sep) # +1 for the space

        # Reverse to original order and join
        prompt = self.asr.sep.join(reversed(prompt_pieces))

        # Context (non_prompt): committed text within current buffer
        context_segments = self.commited[k:]
        context = self.asr.sep.join([t for _, _, t in context_segments])

        return prompt, context

    def process_iter(self):
        """
        Runs on the current audio buffer.

        Returns:
            A tuple (beg_timestamp, end_timestamp, "text"), or (None, None, "").
            The non-emty text is confirmed (committed) partial transcript.
        """

        # get the prompt and context
        prompt, non_prompt = self.prompt()
        logger.debug(f"PROMPT: {prompt}")
        logger.debug(f"CONTEXT: {non_prompt}")
        logger.debug(f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}")

        # transcribe the audio buffer
        # use the prompt to help the model understand the context
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)

        # transform to [(beg, end, "word1"), ...] for alignment
        # it is not necessary to use the prompt, but it helps to get better results
        tsw = self.asr.ts_words(res)

        # using HupothesisBuffer::insert() to deduplicate the repeated words
        self.transcript_buffer.insert(tsw, self.buffer_time_offset)

        # using HupothesisBuffer::flush() to commit the words that are stable
        o = self.transcript_buffer.flush()
        self.commited.extend(o)

        # to_flush(): concatenates the timestamped words or sentences into one sequence that is flushed in one line
        completed = self.to_flush(o)
        logger.debug(f">>>>COMPLETE NOW: {completed}")

        # the rest of the buffer is not commited yet
        the_rest = self.to_flush(self.transcript_buffer.complete())
        logger.debug(f"INCOMPLETE: {the_rest}")

        ### when there is a newly confirmed text
        if o and self.buffer_trimming_way == "sentence":  # trim the completed sentences
            if len(self.audio_buffer) / self.SAMPLING_RATE > self.buffer_trimming_sec:  # longer than this
                self.chunk_completed_sentence()

        if self.buffer_trimming_way == "segment":
            s = self.buffer_trimming_sec  # trim the completed segments longer than s,
        else:
            s = 30 # if the audio buffer is longer than 30s, trim it

        # if the audio buffer is longer than s seconds, trim it
        # default is 30 seconds
        if len(self.audio_buffer) / self.SAMPLING_RATE > s:
            self.chunk_completed_segment(res)

            # alternative: on any word
            #l = self.buffer_time_offset + len(self.audio_buffer)/self.SAMPLING_RATE - 10
            # let's find commited word that is less
            #k = len(self.commited)-1
            #while k>0 and self.commited[k][1] > l:
            #    k -= 1
            #t = self.commited[k][1]
            logger.debug("chunking segment")
            #self.chunk_at(t)

        logger.debug(f"len of buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}")
        # return the committed text: (strat_time, end_time, transcript)
        return self.to_flush(o)

    def chunk_completed_sentence(self):
        """
        Trim the committed buffer based on sentence boundaries,
        ensuring audio and transcript buffers stay within size limits.

        This method is triggered only when:
        - Trimming mode is set to 'sentence'
        - There is enough committed data to identify at least two sentences
        - It then trims the buffer at the boundary between the last two sentences
        """
        if self.commited == []: return
        logger.debug(self.commited)

        # convert the commited words to sentences
        sents = self.words_to_sentences(self.commited)
        for s in sents:
            logger.debug(f"\t\tSENT: {s}")

        # if sentence less than 2, do not trim,
        # otherwise trim the buffer and only keep the last two sentences
        if len(sents) < 2:
            return
        while len(sents) > 2:
            sents.pop(0)

        # we will continue with audio processing at this timestamp
        # Take the end time of the second-to-last sentence as the trimming point
        chunk_at = sents[-2][1]
        logger.debug(f"--- sentence chunked at {chunk_at:2.2f}")
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        """
        Trim audio and transcript buffers based on segment boundaries
        detected from ASR result.

        *** Also ensures chunking happens if audio buffer exceeds a time limit.

        - Uses the second-to-last segment end time (if possible) to trim buffers.
        - Ensures we don't trim beyond the latest committed word's end time.
        - This method is used when buffer_trimming is set to "segment".

        Args:
            res: The ASR model's result from transcribe()
        """
        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE
        if not self.committed:
            if buffer_duration > self.buffer_trimming_sec:
                chunk_time = self.buffer_time_offset + (buffer_duration / 2)
                logger.debug(f"--- No speech detected, forced chunking at {chunk_time:.2f}")
                self.chunk_at(chunk_time)
            return

        logger.debug("Processing committed tokens for segmenting")
        # Get the end timestamps of segments from the ASR result
        # Example: [4.2, 8.7, 12.3] represents the end times of segments
        ends = self.asr.segments_end_ts(res)

        last_committed_time = self.commited[-1][1]
        chunk_done = False

        # If there are at least two segments
        if len(ends) > 1:
            # find a suitable segment boundary `e`
            # why not used the last segment? because it may not commited yet
            # If the segment boundary is later than the end time of the currently confirmed text,
            # it means it has not been confirmed yet and should not be trimmed.
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > last_committed_time:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset

            if e <= last_committed_time:
                logger.debug(f"--- segment chunked at {e:2.2f}")
                self.chunk_at(e)
                chunk_done = True
            else:
                logger.debug(f"--- last segment not within commited area")
        else:
            logger.debug(f"--- not enough segments to chunk")

        if not chunk_done and buffer_duration > self.buffer_trimming_sec:
            logger.debug(f"--- Buffer too large, chunking at last committed time {last_committed_time:.2f}")
            self.chunk_at(last_committed_time)

        logger.debug("Segment chunking complete")

    def chunk_at(self, time):
        """
        Trims the hypothesis buffer and audio buffer at specified timestamp "time".

        Args:
        time (float): The absolute timestamp (in seconds) to trim everything before it.
                      It becomes the new buffer starting point.
        """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset # The length of time to remove from the buffer.
        self.audio_buffer = self.audio_buffer[int(cut_seconds * self.SAMPLING_RATE):]
        self.buffer_time_offset = time # Update the buffer time offset to the new start point.

    def words_to_sentences(self, words):
        """
        Segments a list of timestamped words into sentences using the provided tokenizer.

        Args:
            words (list of (start_time, end_time, word)):
                The list of recognized words with timestamps.

        Returns:
            list of (start_time, end_time, sentence):
                Each item represents a sentence and its time span.

        Example:
            words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world"), (1.0, 1.5, "How"), (1.5, 2.0, "are"), (2.0, 2.5, "you")]
            tokenizer.split() returns: ["Hello world", "How are you"]
            Return: [(0.0, 1.0, "Hello world"), (1.0, 2.5, "How are you")]
        """
        cwords = [w for w in words] # Copy the words to avoid modifying the original list
        t = " ".join(o[2] for o in cwords) # Concatenate the words into a single string
        s = self.tokenizer.split(t) # Split the concatenated string into sentences

        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent # keep the full sentence text
            while cwords:
                b, e, w = cwords.pop(0)
                w = w.strip()
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg, end, fsent))
                    break
                sent = sent[len(w):].strip()
        return out

    def finish(self):
        """
        Flush the remaining (uncommitted) transcript buffer content when processing ends.

        Returns:
            A tuple (start_time, end_time, "text").
            the same format as self.process_iter()
        """
        # Get the uncommitted words
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logger.debug(f"last, noncommited: {f}")

        # Usually at the end, this audio buffer (self.audio_buffer) will also be considered "processed"
        # and will be flushed. So we need to update the buffer time offset.
        # This is important for the next processing cycle.
        self.buffer_time_offset += len(self.audio_buffer) / 16000
        return f

    def to_flush(self, sents, sep=None, offset=0, ):
        """
        Concatenates timestamped words or sentences into a single output line. (flushed in one line)

        Args:
            sents (list): [(start_time, end_time, text), ...] or [] if empty
            sep (str, optional): Separator used between sentence fragments (default is self.asr.sep).
            offset (float, optional): Time offset to be applied to all timestamps.

        Returns:
            tuple: (start_time, end_time, concatenated_text) or (None, None, "") if input is empty.

        Example:
            sents = [(1.0, 2.5, "Hello"), (2.6, 3.2, "world"), (3.3, 4.0, "!")]
            sep = " "
            offset = 0.0
            Returns: (1.0, 4.0, "Hello world")
        """
        # If sep (the separator between sentences/words) is not specified,
        # the sep preset by the ASR model is used (it may be " " or "").
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b, e, t)


class VACOnlineASRProcessor(OnlineASRProcessor):
    '''Wraps OnlineASRProcessor with VAC (Voice Activity Controller).

    It works the same way as OnlineASRProcessor: it receives chunks of audio (e.g. 0.04 seconds),
    it runs VAD and continuously detects whether there is speech or not.
    When it detects end of speech (non-voice for 500ms), it makes OnlineASRProcessor to end the utterance immediately.
    '''

    def __init__(self, online_chunk_size, *a, **kw):
        self.online_chunk_size = online_chunk_size

        self.online = OnlineASRProcessor(*a, **kw)

        # VAC:
        import torch
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        from silero_vad_iterator import FixedVADIterator
        # we use the default options there: 500ms silence, 100ms padding, etc.
        self.vac = FixedVADIterator(model)

        self.logfile = self.online.logfile
        self.init()

    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0  # 目前累積多少 sample 給 ASR。

        self.is_currently_final = False

        self.status = None  # or "voice" or "nonvoice"
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset = 0  # in frames (追蹤整個 session 已經處理到第幾個 frame。)

    def clear_buffer(self):
        """Clears the audio buffer and updates the buffer offset."""
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([], dtype=np.float32)

    def insert_audio_chunk(self, audio):
        res = self.vac(audio)
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is not None:
            # frame is local buffer
            frame = list(res.values())[0] - self.buffer_offset

            # 語音開始
            if 'start' in res and 'end' not in res:
                self.status = 'voice'
                send_audio = self.audio_buffer[frame:]
                # 新的一句話
                self.online.init(offset=(frame + self.buffer_offset)/self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio) # 把從 start 起點以後的音訊送進 ASR
                self.current_online_chunk_buffer_size += len(send_audio)
                self.clear_buffer()
            # 語音結束
            elif 'end' in res and 'start' not in res:
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[:frame]
                self.online.insert_audio_chunk(send_audio) # 把到 end 點為止的音訊送進 ASR
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
            # 如果是 start 和 end 同時都有 (整個語音段)
            else:
                beg = res["start"] - self.buffer_offset
                end = res["end"] - self.buffer_offset
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[beg:end] # 把整段送進 ASR
                self.online.init(offset=(beg+self.buffer_offset)/self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
        # 表示還在持續偵測中
        else:
            if self.status == 'voice':
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()
            else:
                # We keep 1 second because VAD may later find start of voice in it.
                # But we trim it to prevent OOM.
                self.buffer_offset += max(0, len(self.audio_buffer)-self.SAMPLING_RATE)
                self.audio_buffer = self.audio_buffer[-self.SAMPLING_RATE:]

    def process_iter(self):
        if self.is_currently_final:
            return self.finish()
        elif self.current_online_chunk_buffer_size > (self.SAMPLING_RATE * self.online_chunk_size):
            self.current_online_chunk_buffer_size = 0
            ret = self.online.process_iter()
            return ret
        else:
            print("no online update, only VAD", self.status, file=self.logfile)
            return (None, None, "")

    def finish(self):
        ret = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return ret



WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(",")

def create_tokenizer(lan):
    """returns an object that has split function that works like the one of MosesTokenizer"""

    assert lan in WHISPER_LANG_CODES, "language must be Whisper's supported lang code: " + " ".join(WHISPER_LANG_CODES)

    if lan == "uk":
        import tokenize_uk
        class UkrainianTokenizer:
            def split(self, text):
                return tokenize_uk.tokenize_sents(text)
        return UkrainianTokenizer()

    # supported by fast-mosestokenizer
    if lan in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh".split():
        from mosestokenizer import MosesTokenizer
        return MosesTokenizer(lan)

    # the following languages are in Whisper, but not in wtpsplit:
    if lan in "as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt".split():
        logger.debug(f"{lan} code is not supported by wtpsplit. Going to use None lang_code option.")
        lan = None

    from wtpsplit import WtP
    # downloads the model from huggingface on the first use
    wtp = WtP("wtp-canine-s-12l-no-adapters")
    class WtPtok:
        def split(self, sent):
            return wtp.split(sent, lang_code=lan)
    return WtPtok()


def add_shared_args(parser):
    """shared args for simulation (this entry point) and server
    parser: argparse.ArgumentParser object
    """
    parser.add_argument('--min-chunk-size', type=float, default=1.0, help='Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.')
    parser.add_argument('--model', type=str, default='large-v2', choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo".split(","),help="Name size of the Whisper model to use (default: large-v2). The model is automatically downloaded from the model hub if not present in model cache dir.")
    parser.add_argument('--model_cache_dir', type=str, default=None, help="Overriding the default model cache dir where models downloaded from the hub are saved")
    parser.add_argument('--model_dir', type=str, default=None, help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.")
    parser.add_argument('--lan', '--language', type=str, default='auto', help="Source language code, e.g. en,de,cs, or 'auto' for language detection.")
    parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe","translate"],help="Transcribe or translate.")
    parser.add_argument('--backend', type=str, default="faster-whisper", choices=["faster-whisper", "whisper_timestamped", "mlx-whisper", "openai-api"],help='Load only this backend for Whisper processing.')
    parser.add_argument('--vac', action="store_true", default=False, help='Use VAC = voice activity controller. Recommended. Requires torch.')
    parser.add_argument('--vac-chunk-size', type=float, default=0.04, help='VAC sample size in seconds.')
    parser.add_argument('--vad', action="store_true", default=False, help='Use VAD = voice activity detection, with the default parameters.')
    parser.add_argument('--buffer_trimming', type=str, default="segment", choices=["sentence", "segment"],help='Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.')
    parser.add_argument('--buffer_trimming_sec', type=float, default=15, help='Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.')
    parser.add_argument("-l", "--log-level", dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the log level", default='DEBUG')

def asr_factory(args, logfile=sys.stderr):
    """
    Creates and configures an ASR and ASR Online instance based on the specified backend and arguments.
    """
    backend = args.backend
    if backend == "openai-api":
        logger.debug("Using OpenAI API.")
        asr = OpenaiApiASR(lan=args.lan)
    else:
        if backend == "faster-whisper":
            asr_cls = FasterWhisperASR
        elif backend == "mlx-whisper":
            asr_cls = MLXWhisper
        else:
            asr_cls = WhisperTimestampedASR

        # Only for FasterWhisperASR and WhisperTimestampedASR
        size = args.model
        t = time.time()
        logger.info(f"Loading Whisper {size} model for {args.lan}...")
        asr = asr_cls(modelsize=size, lan=args.lan, cache_dir=args.model_cache_dir, model_dir=args.model_dir)
        e = time.time()
        logger.info(f"done. It took {round(e-t,2)} seconds.")

    # Apply common configurations
    if getattr(args, 'vad', False):  # Checks if VAD argument is present and True
        logger.info("Setting VAD filter")
        asr.use_vad()

    language = args.lan
    if args.task == "translate":
        asr.set_translate_task()
        tgt_language = "en"  # Whisper translates into English
    else:
        tgt_language = language  # Whisper transcribes in this language

    # Create the tokenizer
    if args.buffer_trimming == "sentence":
        tokenizer = create_tokenizer(tgt_language)
    else:
        tokenizer = None

    # Create the OnlineASRProcessor
    if args.vac:

        online = VACOnlineASRProcessor(args.min_chunk_size, asr,tokenizer,logfile=logfile,buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))
    else:
        online = OnlineASRProcessor(asr,tokenizer,logfile=logfile,buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))

    return asr, online

def set_logging(args,logger,other="_server"):
    logging.basicConfig(#format='%(name)s
            format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)
    logging.getLogger("whisper_online"+other).setLevel(args.log_level)
#    logging.getLogger("whisper_online_server").setLevel(args.log_level)



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str, help="Filename of 16kHz mono channel wav, on which live streaming is simulated.")
    add_shared_args(parser)
    parser.add_argument('--start_at', type=float, default=0.0, help='Start processing audio at this time.')
    parser.add_argument('--offline', action="store_true", default=False, help='Offline mode.')
    parser.add_argument('--comp_unaware', action="store_true", default=False, help='Computationally unaware simulation.')

    args = parser.parse_args()

    # reset to store stderr to different file stream, e.g. open(os.devnull,"w")
    logfile = sys.stderr

    if args.offline and args.comp_unaware:
        logger.error("No or one option from --offline and --comp_unaware are available, not both. Exiting.")
        sys.exit(1)

#    if args.log_level:
#        logging.basicConfig(format='whisper-%(levelname)s:%(name)s: %(message)s',
#                            level=getattr(logging, args.log_level))

    set_logging(args,logger)

    audio_path = args.audio_path

    SAMPLING_RATE = 16000
    duration = len(load_audio(audio_path))/SAMPLING_RATE
    logger.info("Audio duration is: %2.2f seconds" % duration)

    asr, online = asr_factory(args, logfile=logfile)
    if args.vac:
        min_chunk = args.vac_chunk_size
    else:
        min_chunk = args.min_chunk_size

    # load the audio into the LRU cache before we start the timer
    a = load_audio_chunk(audio_path,0,1)

    # warm up the ASR because the very first transcribe takes much more time than the other
    asr.transcribe(a)

    beg = args.start_at
    start = time.time()-beg

    def output_transcript(o, now=None):
        # output format in stdout is like:
        # 4186.3606 0 1720 Takhle to je
        # - the first three words are:
        #    - emission time from beginning of processing, in milliseconds
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if now is None:
            now = time.time()-start
        if o[0] is not None:
            print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),file=logfile,flush=True)
            print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),flush=True)
        else:
            # No text, so no output
            pass

    if args.offline: ## offline mode processing (for testing/debugging)
        a = load_audio(audio_path)
        online.insert_audio_chunk(a)
        try:
            o = online.process_iter()
        except AssertionError as e:
            logger.error(f"assertion error: {repr(e)}")
        else:
            output_transcript(o)
        now = None
    elif args.comp_unaware:  # computational unaware mode
        end = beg + min_chunk
        while True:
            a = load_audio_chunk(audio_path,beg,end)
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {repr(e)}")
                pass
            else:
                output_transcript(o, now=end)

            logger.debug(f"## last processed {end:.2f}s")

            if end >= duration:
                break

            beg = end

            if end + min_chunk > duration:
                end = duration
            else:
                end += min_chunk
        now = duration

    else: # online = simultaneous mode
        end = 0
        while True:
            now = time.time() - start
            if now < end+min_chunk:
                time.sleep(min_chunk+end-now)
            end = time.time() - start
            a = load_audio_chunk(audio_path,beg,end)
            beg = end
            online.insert_audio_chunk(a)

            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {e}")
                pass
            else:
                output_transcript(o)
            now = time.time() - start
            logger.debug(f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}")

            if end >= duration:
                break
        now = None

    o = online.finish()
    output_transcript(o, now=now)
