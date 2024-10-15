from collections.abc import Sequence
import concurrent.futures
from typing import Any, Callable, Dict

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import clock as game_clock
from concordia.typing import component
from concordia.utils import measurements as measurements_lib
import numpy as np


class UtilitarianGoalAchievement(component.Component):
  """Utilitarian Goal Achievement component that tracks goal progress based on utilities."""

  def __init__(
    self,
    model: language_model.LanguageModel,
    player_name: str,
    player_goal: str,
    clock: game_clock.GameClock,
    init_context_fn: Callable[[], str],
    context_fn: Callable[[], str],
    name: str = 'Utilitarian Goal Achievement',
    measurements: measurements_lib.Measurements | None = None,
    channel: str = 'utilitarian_goal_achievement',
    verbose: bool = False,
  ):
    """Initializes the metric.

    Args:
      model: Language model to use for the question.
      player_name: Name of the player.
      player_goal: Player's main goal.
      clock: Clock for logging and tracking time.
      context_fn: Function that provides the context (e.g., game state).
      name: Name of the component.
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

    # Ask the model for a baseline utility item (numéraire)
    numeraire_question = (
      "What is the most important item or action that should be used as a baseline for utility calculations in this scenario?"
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

    # Ask the model for several grounding utilities for comparison
    grounding_question = (
      f"List the key quantifiable events or items in this environment "
      f"and assign a utility value in units of the numéraire."
      f"For example, if the numéraire was 'dollars', and some important items"
      f"in your environment are gold, apples, and potatoes, your output"
      f"should be formatted as follows:\n"
      f"dollars: 1.0\n"
      f"gold: 100.0\n"
      f"potatoes: 3.0\n"
      f"apples: 2.5\n"
      f"Your numeraire is '{self._numeraire}'."
    )
    grounding_response = prompt.open_question(
      question=grounding_question,
      answer_prefix=f"{self._numeraire}: 1.0\n",
      answer_suffix=""
    )
    # append the numeraire to the response
    grounding_response = f"{self._numeraire}: 1.0\n" + grounding_response
    if self._verbose:
      print(f"Grounding utilities response: {grounding_response}")

    self._grounding_response = grounding_response

    # Parse the response into a dictionary of utilities (assume well-formed)
    utilities = self._parse_grounding_utilities(grounding_response)
    return utilities

  def _parse_grounding_utilities(self, response: str) -> Dict[str, float]:
    """Parses the response from the LLM into a dictionary of utilities.
    We don't currently used these parsed values, instead opting
    to work with the raw response string, but in the future we may
    want to use these values for some other purpose.

    Args:
      response: The response string from the LLM.
    """
    utilities = {}
    lines = response.split("\n")
    for line in lines:
      try:
        item, value = line.split(":")
        utilities[item.strip()] = float(value.strip())
      except ValueError:
        continue  # Skip lines that don't fit the format
    return utilities

  def name(self) -> str:
    """Returns the name of the component."""
    return self._name

  def _calculate_utility(self, action: str) -> float:
    """Queries the LLM to calculate the utility of a given action."""
    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(self._context_fn())

    # Construct the question asking the model to show its work
    question = (
      f"We will now calculate the utility of a specific action. Example:"
      f"if you value gold at 100.0 dollars, apples at 2.5 dollars, and "
      f"potatoes at 3.0 dollars, "
      f"and the numéraire is '{self._numeraire}', and your action is mine gold, "
      f"you might decide that the utility of mining gold is 100.0, so you would "
      f"write {{100.0}} in the response box. If your action was instead 'buy an "
      f"apple for 2.3 dollars,' you might decide that the utility of buying an "
      f"apple is 2.5, and you pay 2.3 dollars for it, so because 2.5 - 2.3 = 0.2, "
      f"you would write {{0.2}} in the response box.\n"
      f"Your numéraire is '{self._numeraire}', and your grounding utilities are:\n"
      f"{self._grounding_response}\n"
      f"Please calculate the utility of the action '{action}'.\n"
      f"Show your work and return the final utility within double curly braces."
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

  def update_after_event(self, action: str) -> None:
    """Update utility after a specific action occurs."""
    utility_value = self._calculate_utility(action)

    datum = {
      'time_str': self._clock.now().strftime('%H:%M:%S'),
      'clock_step': self._clock.get_step(),
      'timestep': self._timestep,
      'utility_value': utility_value,
      'player': self._player_name,
      'goal': self._player_goal,
    }
    datum['time'] = self._clock.now()

    if self._measurements:
      self._measurements.publish_datum(self._channel, datum)
    if self._verbose:
      print(f"{self._name} for player '{self._player_name}' on action '{action}': Utility = {utility_value}")

    self._timestep += 1

  def state(self) -> str | None:
    """Returns the current state of the component"""
    return ''
