from collections.abc import Sequence
from typing import Callable, Dict

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import clock as game_clock
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.utils import helper_functions
from concordia.utils import measurements as measurements_lib
import numpy as np
from typing_extensions import override


class UtilitarianActingComponent(entity_component.ActingComponent):
  """A Utilitarian Acting Component that tracks goal progress and performs actions based on utility."""

  def __init__(
    self,
    model: language_model.LanguageModel,
    player_name: str,
    player_goal: str,
    clock: game_clock.GameClock,
    init_context_fn: Callable[[], str],
    context_fn: Callable[[], str],
    name: str = 'Utilitarian Acting Component',
    measurements: measurements_lib.Measurements | None = None,
    channel: str = 'utilitarian_goal_achievement',
    verbose: bool = False,
  ):
    """Initializes the acting component.

    Args:
      model: Language model to use for generating actions.
      player_name: Name of the player.
      player_goal: Player's goal to be achieved.
      clock: Clock for logging and tracking time.
      init_context_fn: Function that provides the initial context (e.g., game state).
      context_fn: Function that provides the current context (e.g., game state).
      name: Name of the acting component.
      measurements: The measurements object to publish data to.
      channel: Channel to use for logging the metric.
      verbose: Whether to print logs during execution.
    """
    self._model = model
    self._player_name = player_name
    self._player_goal = player_goal
    self._clock = clock
    self._init_context_fn = init_context_fn
    self._context_fn = context_fn
    self._name = name
    self._measurements = measurements
    self._channel = channel
    self._verbose = verbose
    self._timestep = 0

    # Initialize numéraire and grounding utilities
    self._numeraire = self._initialize_numeraire()
    self._grounding_utilities = self._initialize_grounding_utilities()

  def _initialize_numeraire(self) -> str:
    """Initialize the numéraire by querying the LLM for a baseline item."""
    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(self._init_context_fn())

    numeraire_question = (
      "What is the most important item or action that should be used as a baseline for utility calculations?"
    )
    numeraire = prompt.open_question(
      question=numeraire_question,
      answer_prefix="The numéraire is ",
      answer_suffix="."
    )
    if self._verbose:
      print(f"Numéraire initialized as: {numeraire}")
    return numeraire.strip()

  def _initialize_grounding_utilities(self) -> Dict[str, float]:
    """Initialize grounding utilities for common events or items."""
    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(self._init_context_fn())

    grounding_question = (
      f"List the key quantifiable events or items in this environment "
      f"and assign a utility value in units of the numéraire."
      f"Your numeraire is '{self._numeraire}'."
    )
    grounding_response = prompt.open_question(
      question=grounding_question,
      answer_prefix=f"{self._numeraire}: 1.0\n",
      answer_suffix=""
    )
    grounding_response = f"{self._numeraire}: 1.0\n" + grounding_response
    if self._verbose:
      print(f"Grounding utilities response: {grounding_response}")

    utilities = self._parse_grounding_utilities(grounding_response)
    return utilities

  def _parse_grounding_utilities(self, response: str) -> Dict[str, float]:
    """Parses the response from the LLM into a dictionary of utilities."""
    utilities = {}
    lines = response.split("\n")
    for line in lines:
      try:
        item, value = line.split(":")
        utilities[item.strip()] = float(value.strip())
      except ValueError:
        continue  # Skip lines that don't fit the format
    return utilities

  def _calculate_utility(self, action: str) -> float:
    """Queries the LLM to calculate the utility of a given action."""
    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(self._context_fn())

    question = (
      f"Calculate the utility of the action '{action}' using the numéraire '{self._numeraire}' "
      f"and the grounding utilities:\n{self._grounding_utilities}\n"
      f"Show your work and return the final utility value within double curly braces."
    )
    response = prompt.open_question(question=question)
    if self._verbose:
      print(f"Utility calculation response: {response}")

    # Extract the final utility value within {{utility_value}} delimiters
    start = response.find("{{")
    end = response.find("}}")
    if start != -1 and end != -1:
      utility_value = float(response[start + 2:end].strip())
      return utility_value
    else:
      print("Failed to extract utility value from response.")
      return 0.0

  @override
  def get_action_attempt(
    self,
    contexts: entity_component.ComponentContextMapping,
    action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Sample multiple actions and select the one with the highest utility."""

    prompt = interactive_document.InteractiveDocument(self._model)
    context = self._context_fn()
    prompt.statement(context)

    call_to_action = action_spec.call_to_action.format(
      name=self._player_name,
      timedelta=helper_functions.timedelta_to_readable_str(
        self._clock.get_step_size()
      ),
    )

    # For FREE output type, generate multiple action candidates and pick the one with the highest utility
    if action_spec.output_type == entity_lib.OutputType.FREE:
      sampled_actions = []
      sampled_utilities = []

      # Sample multiple actions by querying the model multiple times
      for _ in range(5):  # Sample 5 actions (you can adjust this number)
        action = self.get_entity().name + ' '
        action += prompt.open_question(
            call_to_action,
            max_tokens=2200,
            answer_prefix=action,
            # This terminator protects against the model providing extra context
            # after the end of a directly spoken response, since it normally
            # puts a space after a quotation mark only in these cases.
            terminators=('" ', '\n'),
            question_label='Exercise',
        )
        self._log(action, prompt)
        sampled_actions.append(action)

        # Calculate utility for the sampled action
        utility_value = self._calculate_utility(action)
        sampled_utilities.append(utility_value)

      # Find the action with the maximum utility
      max_index = np.argmax(sampled_utilities)
      best_action = sampled_actions[max_index]
      best_utility = sampled_utilities[max_index]

      result = f"Best Action: {best_action}, Utility: {best_utility}"
      self._log(result, prompt)
      return result

    # For CHOICE output type, evaluate the utilities of the predefined choices
    elif action_spec.output_type == entity_lib.OutputType.CHOICE:
      sampled_utilities = []

      # Calculate the utility for each predefined choice
      for option in action_spec.options:
        utility_value = self._calculate_utility(option)
        sampled_utilities.append(utility_value)

      # Find the option with the maximum utility
      max_index = np.argmax(sampled_utilities)
      best_option = action_spec.options[max_index]
      best_utility = sampled_utilities[max_index]

      result = f"Best Action: {best_option}, Utility: {best_utility}"
      self._log(result, prompt)
      return best_option

    # For FLOAT output type, sample actions and choose the one that makes sense as a float
    elif action_spec.output_type == entity_lib.OutputType.FLOAT:
      sampled_actions = []
      sampled_utilities = []

      # Sample multiple actions and parse them as floats
      for _ in range(5):  # Sample 5 float values (you can adjust this number)
        prefix = self.get_entity().name + ' '
        sampled_text = prompt.open_question(
            call_to_action,
            max_tokens=2200,
            answer_prefix=prefix,
        )
        self._log(sampled_text, prompt)
        try:
          sampled_text =  str(float(sampled_text))
        except ValueError:
          sampled_text =  '0.0'

        sampled_actions.append(sampled_text)
        sampled_utilities.append(self._calculate_utility(str(sampled_text)))

      # Find the float with the maximum utility
      max_index = np.argmax(sampled_utilities)
      best_float = sampled_actions[max_index]
      best_utility = sampled_utilities[max_index]

      result = f"Best Float Action: {best_float}, Utility: {best_utility}"
      self._log(result, prompt)
      return str(best_float)

    else:
      raise NotImplementedError(
        f"Unsupported output type: {action_spec.output_type}. "
        "Supported output types are: FREE, CHOICE, and FLOAT."
      )

  def _log(self, result: str, prompt: interactive_document.InteractiveDocument):
    if self._measurements:
      self._measurements.publish_datum(self._channel, {
        'Key': self._name,
        'Value': result,
        'Prompt': prompt.view().text().splitlines(),
      })
