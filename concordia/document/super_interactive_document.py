"""Utilities for chain-of-thought prompting."""

from collections.abc import Collection, Iterable, Iterator, Sequence

from concordia.document.interactive_document import InteractiveDocument

DEFAULT_MAX_CHARACTERS = 200
DEFAULT_MAX_TOKENS = DEFAULT_MAX_CHARACTERS // 4

DEBUG_TAG = 'debug'
STATEMENT_TAG = 'statement'
QUESTION_TAG = 'question'
RESPONSE_TAG = 'response'
MODEL_TAG = 'model'
INTERACTIVE_TAGS = frozenset(
    {DEBUG_TAG, STATEMENT_TAG, QUESTION_TAG, RESPONSE_TAG, MODEL_TAG}
)


class SuperInteractiveDocument(InteractiveDocument):
    """A subclass of InteractiveDocument with additional functionality."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Initializes the instance."""
        super().__init__(*args, **kwargs)

    def super_open_question(
        self,
        question: str,
        *,
        temperature: float = 1.0,  # Additional temperature argument
        forced_response: str | None = None,
        answer_prefix: str = '',
        answer_suffix: str = '',
        max_tokens: int = DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = ('\n',),
        question_label: str = 'Question',
        answer_label: str = 'Answer',
    ) -> str:
        """Asks the agent an open question with a specified temperature and appends it to the document."""
        self._question(f'{question_label}: {question}\n')
        self._response(f'{answer_label}: {answer_prefix}')

        if forced_response is None:
            response = self._model.sample_text(
                prompt=self._model_view.text(),
                max_tokens=max_tokens,
                terminators=terminators,
                temperature=temperature  # Use the temperature argument
            )
        else:
            response = forced_response

        response = response.removeprefix(answer_prefix)
        self._model_response(response)
        self._response(f'{answer_suffix}\n')
        return response

    def super_open_question_multiple(
        self,
        question: str,
        *,
        num_samples: int = 3,  # Number of responses to sample
        temperature: float = 1.0,  # Sampling temperature
        forced_response: str | None = None,
        answer_prefix: str = '',
        answer_suffix: str = '',
        max_tokens: int = DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = ('\n',),
        question_label: str = 'Question',
        answer_label: str = 'Answer',
    ) -> list[str]:
        """Asks the agent an open question and samples multiple responses.

        Args:
            question: The question to ask.
            num_samples: The number of responses to sample from the model.
            temperature: Sampling temperature for the LLM.
            forced_response: Forces the document to provide this response.
            answer_prefix: A prefix to append to the model's prompt.
            answer_suffix: A suffix to append to the model's response.
            max_tokens: The maximum number of tokens to sample from the model.
            terminators: Strings that must not be present in the model's response.
            question_label: The label to use for the question, typically "Question".
            answer_label: The label to use for the answer, typically "Answer".

        Returns:
            A list of sampled responses (or a single `forced_response` if provided).
        """
        self._question(f'{question_label}: {question}\n')
        self._response(f'{answer_label}: {answer_prefix}')

        if forced_response is None:
            # Modify the model call to generate multiple samples
            responses = self._model.sample_text(
                prompt=self._model_view.text(),
                max_tokens=max_tokens,
                num_samples=num_samples,  # Sample multiple responses
                temperature=temperature,
                terminators=terminators,
            )
        else:
            responses = [forced_response] * num_samples

        # Clean up responses by removing the prefix if needed
        cleaned_responses = [response.removeprefix(answer_prefix).strip() for response in responses]

        # Append the first response to the document and log all responses
        self._model_response(cleaned_responses[0])
        self._response(f'{answer_suffix}\n')

        # Log additional responses if verbose is set
        if self._verbose:
            for i, response in enumerate(cleaned_responses, start=1):
                print(f"Sampled Response {i}: {response}")

        return cleaned_responses
