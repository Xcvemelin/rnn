import os
import heapq
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Union
from configs.logging import logger
from pathlib import Path
from configs.paths import VISUALIZATIONS_DIR

class TextGenerator():
    ''' A class for text generation '''
    def __init__(self, char2idx: Dict[str, int], idx2char: Dict[int, str]):
        ''' Initializing TextGenerator variables '''
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.rng = np.random.default_rng()
        logger.debug("TextGenerator initialized")

    def _greedy_sample(self, logits):
        return tf.argmax(logits).numpy()

    def _top_k_sample(self, logits, k):
        values, indices = tf.math.top_k(logits, k=k)
        # Convert to numpy and apply softmax to get probabilities
        values = values.numpy()
        probs = np.exp(values) / np.sum(np.exp(values)) # softmax
        # Sample from the top_k candidates
        return self.rng.choice(indices.numpy(), p=probs)

    def _random_sample(self, logits):
        return tf.random.categorical(tf.expand_dims(logits, 0), num_samples=1)[0, 0].numpy()

    def with_sampling(self, model, start_string, num_generate=300, temperature=1.0, method='top_k', k=5):
        ''' Text generation with sampling method: greedy, top_k, or random '''
        input_eval = [self.char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        for _ in range(num_generate):
            predictions = model(input_eval)
            predictions = predictions[:, -1, :] / temperature
            predicted_logits = tf.squeeze(predictions, 0)

            match method:
                case 'greedy':
                    predicted_id = self._greedy_sample(predicted_logits)
                case 'top_k':
                    predicted_id = self._top_k_sample(predicted_logits, k)
                case _:
                    predicted_id = self._random_sample(predicted_logits)

            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(self.idx2char[predicted_id])

        return start_string + ''.join(text_generated)

    def beam_search(self, model, start_string, num_generate=50, beam_width=3, temperature=1.0):
        ''' Text generation using beam search '''
        input_eval = [self.char2idx[s] for s in start_string]
        beams = [(start_string, input_eval, 0.0)]

        MAX_CONTEXT_LENGTH = 100  # Limit context length for efficiency

        for _ in range(num_generate):
            all_candidates = []

            for text_so_far, seq, score in beams:
                input_seq = tf.expand_dims(seq[-MAX_CONTEXT_LENGTH:], 0)
                predictions = model(input_seq)
                predictions = predictions[:, -1, :] / temperature
                probs = tf.nn.softmax(predictions).numpy().flatten()

                top_indices = np.argpartition(probs, -beam_width)[-beam_width:]
                for idx in top_indices:
                    prob = probs[idx]
                    candidate_text = text_so_far + self.idx2char[idx]
                    candidate_seq = seq + [idx]
                    candidate_score = score + np.log(prob + 1e-9)
                    all_candidates.append((candidate_text, candidate_seq, candidate_score))

            beams = heapq.nlargest(beam_width, all_candidates, key=lambda x: x[2])

        return beams[0][0]

    def visualize_logits_probs(self, model, input_text, temperature=1.0, top_k=5, path='outputs/visualizations/visualize.png'):
        ''' Visualize logits and probabilities for a single generation step '''
        input_eval = [self.char2idx[s] for s in input_text]
        input_eval = tf.expand_dims(input_eval, 0)

        predictions = model(input_eval)
        logits = predictions[:, -1, :].numpy().flatten()

        scaled_logits = logits / temperature

        # Получаем вероятности через softmax
        probs = tf.nn.softmax(scaled_logits).numpy()

        # Top-k индексов по вероятности
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_logits = logits[top_indices]
        top_probs = probs[top_indices]
        top_chars = [self.idx2char[i] for i in top_indices]

        # Визуализация
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        axs[0].bar(top_chars, top_logits, color='skyblue')
        axs[0].set_title("Top Logits (до Softmax)")
        axs[0].set_ylabel("Логит")

        axs[1].bar(top_chars, top_probs, color='orange')
        axs[1].set_title("Top Probabilities (после Softmax)")
        axs[1].set_ylabel("Вероятность")

        for ax in axs:
            ax.set_xlabel("Символы")
            ax.grid(True)

        plt.suptitle(f"Следующий символ после: '{input_text}' (temperature={temperature})")
        plt.tight_layout()

        visualizations_dir = VISUALIZATIONS_DIR / 'visualize.png'
        path = Path(visualizations_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        logger.success("Saved in outputs/visualizations/")