from collections.abc import Sequence
import concurrent.futures
from typing import Any, Callable, Dict

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import clock as game_clock
from concordia.typing import component
from concordia.utils import measurements as measurements_lib
import numpy as np

DEFAULT_SCALE_TO_FLOAT = {
  "certainly": 1.0,
  "very likely": 0.99,
  "likely": 0.8,
  "unlikely": 0.2,
  "very unlikely": 0.01,
  "impossible": 0.001,
}

DEFAULT_CHANNEL_NAME = 'model_of_others'

DEFAULT_NUM_ARCHETYPES = 20

class ModelOfOthersMetric(component.Component):
  """Maintains Bayesian priors over the state of other players."""

  def __init__(
    self,
    *,
    model: language_model.LanguageModel,
    player_name: str,
    player_names: Sequence[str],
    context_fn: Callable[[], str],
    clock: game_clock.GameClock,
    name: str = 'ModelOfOthers',
    scale_to_float: Dict[str, float] = DEFAULT_SCALE_TO_FLOAT,
    verbose: bool = False,
    measurements: measurements_lib.Measurements | None = None,
    channel: str = DEFAULT_CHANNEL_NAME,
    num_archetypes: int = DEFAULT_NUM_ARCHETYPES,
  ):
    """
    Initialize the model, generate possible player archetypes using LLM.

    Args:
      model: Language model to use for the question prompts.
      player_names: List of player names.
      context_fn: Function that provides the context (e.g., game state).
      clock: Clock object to track time and log data.
      verbose: If True, logs details about the model's execution.
    """
    self._model = model
    self._name = name
    self._clock = clock
    self._verbose = verbose
    self._player_name = player_name
    if player_names:
      self._player_names = list(player_names)
    else:
      raise ValueError('player_names must be specified.')
    self._context_fn = context_fn
    if scale_to_float:
      self._scale_to_float = scale_to_float
    else:
      raise ValueError('scale must be specified.')
    self._measurements = measurements
    self._channel = channel

    if self._measurements:
      self._measurements.get_channel(self._channel)

    self._timestep = 0

    # Number of possible player archetypes
    self._num_archetypes = num_archetypes

    # Dictionary to store priors for each player (uniform priors initially)
    self._priors: Dict[str, np.ndarray] = {
      player: np.ones(self._num_archetypes) / self._num_archetypes for player in self._player_names
    }

    # Generate the list of possible player types via LLM.
    self._archetypes = self._generate_player_archetypes()

  def _generate_player_archetypes(self) -> Sequence[str]:
    """
    Queries the LLM to generate self.num_archetypes distinct player types for the scenario.
    """
    template_1 = "Describe the sort of player that would {verb} {agent_name} in this situation."

    template_1_verbs = [
      "benefit",
      "hurt",
      "help",
      "harm",
      "support",
      "oppose",
      "assist",
      "disrupt",
      "upset",
      "please",
      "teach",
      "learn from",
      "play with",
      "attack",
      "purchase from",
      "sell to",
      "trade with",
      "cure"
    ]

    template_2 = "Describe an extremely {adjective} player in this situation."
    template_2_adjectives = [
      "peaceful",
      "smart",
      "confusing",
      "irrelevant",
      "cautious",
      "reckless",
      "manipulative",
      "greedy",
      "stingy",
      "generous",
      "kind",
      "cruel",
      "helpful",
      "harmful",
      "friendly",
      "hostile",
      "cooperative",
      "competitive",
      "collaborative",
      "sneaky",
      "honest",
      "dishonest",
      "loyal",
      "disloyal",
      "trustworthy",
      "untrustworthy",
      "brave",
      "cowardly",
    ]

    prompts = []
    for verb in template_1_verbs:
      prompts.append(template_1.format(verb=verb, agent_name=self._player_name))
    for adjective in template_2_adjectives:
      prompts.append(template_2.format(adjective=adjective))


    # Use the LLM to generate a list of player types.
    archetypes = []
    for p in prompts:
      prompt = interactive_document.InteractiveDocument(self._model)
      prompt.statement(self._context_fn())
      response = prompt.open_question(question=p)
      archetypes.append(response.strip())

    # select num_archetypes from the generated archetypes at random.
    archetypes = np.random.choice(archetypes, self._num_archetypes, replace=False)

    if self._verbose:
      print("Generated player archetypes:", archetypes)

    return archetypes

  def _get_action_likelihood(self, player: str) -> np.ndarray:
    """
    Queries the LLM to get the likelihood that a given player type would have
    led to the observed action.

    Args:
      player: The name of the player performing the action.

    Returns:
      A probability distribution over the archetypes for this action.
    """

    # Query the LLM with a multiple-choice question.
    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(self._context_fn())

    question = "How likely is the following description of {player}?\nDescription: {archetype}"

    # Assign natural language likelihoods to probabilities
    choices = ["certainly", "very likely", "likely", "unlikely", "very unlikely", "impossible"]
    probabilities = [1, 0.99, 0.8, 0.2, 0.01, 0.001]

    # Ask the model about the action likelihood for each archetype
    likelihoods = np.zeros(self._num_archetypes)
    for i, archetype in enumerate(self._archetypes):
      answer = prompt.multiple_choice_question(
        question=question.format(player=player, archetype=archetype),
        answers=choices,
      )
      likelihoods[i] = probabilities[answer]

    if self._verbose:
      print(f"Likelihoods for player '{player}' performing action:", likelihoods)

    return likelihoods

  def _get_bayesian_update(self, player) -> None:
    """
    Perform Bayesian update for a player based on the action they performed.

    Args:
      player_action: A tuple containing the player's name and their action.
    """

    prior = self._priors[player]
    likelihood = self._get_action_likelihood(player)

    # Perform Bayesian update: Posterior ∝ Prior * Likelihood
    posterior = prior * likelihood
    posterior /= posterior.sum()  # Normalize the posterior

    # Update the player's priors
    self._priors[player] = posterior

    if self._verbose:
      print(f"Updated priors for player '{player}':", posterior)

  def update(self) -> None:
    """See base class."""

    with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(self._player_names)
    ) as executor:
      executor.map(self._get_bayesian_update, self._player_names)
    self._timestep += 1

  def state(
    self,
  ) -> str | None:
    """Returns the current state of the component."""
    return ''

  def get_priors(self) -> Dict[str, np.ndarray]:
    """
    Return the current priors for all players.
    """
    return self._priors
