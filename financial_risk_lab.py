"""
Varda: AI-Powered Financial Risk Lab

Varda is an AI-powered financial risk lab that uses physics-inspired models,
Monte Carlo simulation, and network analytics to visualize how credit and
systemic risk propagate through real-world relationships.

Core Capabilities:
- Fluid dynamics metaphors and network models for risk propagation
- Monte Carlo simulations for scenario analysis
- Markov chain models for state transitions (credit ratings, risk states)
- AI predictors for risk forecasting
- Credit and systemic risk analytics
- Network-based visualization of risk flow across relationships
  (borrowers, lenders, suppliers, etc.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings


class RiskType(Enum):
    """Types of financial risk modeled by Varda."""
    CREDIT = "credit"
    SYSTEMIC = "systemic"
    CONTAGION = "contagion"
    LIQUIDITY = "liquidity"
    MARKET = "market"


@dataclass
class Entity:
    """Represents a financial entity (borrower, lender, supplier, etc.) in the network."""
    id: str
    name: str
    entity_type: str  # e.g., "borrower", "lender", "supplier", "bank"
    initial_risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate entity data."""
        if not 0.0 <= self.initial_risk_score <= 1.0:
            warnings.warn(f"Risk score for {self.id} should be in [0, 1], got {self.initial_risk_score}")


@dataclass
class Relationship:
    """Represents a relationship/connection between entities."""
    source_id: str
    target_id: str
    relationship_type: str  # e.g., "lending", "supply_chain", "derivative"
    strength: float = 1.0  # Connection strength (0-1)
    exposure: float = 0.0  # Financial exposure amount
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate relationship data."""
        if not 0.0 <= self.strength <= 1.0:
            warnings.warn(f"Relationship strength should be in [0, 1], got {self.strength}")


@dataclass
class MarketConstraint:
    """
    Represents a constraint that affects market state transitions.
    
    Constraints can represent:
    - Economic indicators (inflation, unemployment, GDP growth)
    - Policy constraints (interest rates, regulatory limits)
    - Market conditions (volatility, liquidity, credit spreads)
    - External shocks (geopolitical events, natural disasters)
    """
    name: str
    constraint_type: str  # e.g., "economic", "policy", "market", "external"
    value: float  # Current value of the constraint
    impact_on_transitions: Dict[str, float] = field(default_factory=dict)
    # Dict mapping "from_state->to_state" to impact multiplier
    # e.g., {"Normal->Stressed": 1.5, "Stressed->Crisis": 2.0}
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_transition_impact(self, from_state: str, to_state: str) -> float:
        """Get the impact multiplier for a specific transition."""
        key = f"{from_state}->{to_state}"
        return self.impact_on_transitions.get(key, 1.0)


@dataclass
class MarketState:
    """
    Represents a market state/regime as an entity with constraints.
    
    Market states can be:
    - Normal: Stable market conditions
    - Stressed: Elevated risk, some volatility
    - Crisis: Severe market disruption
    - Recovery: Post-crisis stabilization
    """
    state_name: str
    description: str = ""
    base_stability: float = 0.5  # Base probability of staying in this state (0-1)
    constraints: List[MarketConstraint] = field(default_factory=list)
    economic_indicators: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate market state data."""
        if not 0.0 <= self.base_stability <= 1.0:
            warnings.warn(f"Base stability should be in [0, 1], got {self.base_stability}")


class MarkovChain:
    """
    Markov Chain model for financial state transitions.
    
    Useful for modeling:
    - Credit rating transitions (AAA -> AA -> A -> ... -> Default)
    - Risk state evolution (Low -> Medium -> High -> Default)
    - Regime switching (Normal -> Stressed -> Crisis)
    """
    
    def __init__(
        self,
        states: List[str],
        transition_matrix: np.ndarray,
        initial_distribution: Optional[np.ndarray] = None
    ):
        """
        Initialize a Markov Chain.
        
        Args:
            states: List of state names (e.g., ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"])
            transition_matrix: Square matrix where entry (i,j) is P(state_j | state_i)
            initial_distribution: Initial probability distribution over states (defaults to uniform)
        """
        self.states = states
        self.n_states = len(states)
        self.state_to_idx = {state: idx for idx, state in enumerate(states)}
        
        # Validate transition matrix
        transition_matrix = np.array(transition_matrix)
        if transition_matrix.shape != (self.n_states, self.n_states):
            raise ValueError(f"Transition matrix must be {self.n_states}x{self.n_states}")
        
        # Normalize rows to ensure they sum to 1
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.transition_matrix = transition_matrix / row_sums
        
        # Validate probabilities
        if np.any(self.transition_matrix < 0) or np.any(self.transition_matrix > 1):
            raise ValueError("Transition probabilities must be in [0, 1]")
        
        # Set initial distribution
        if initial_distribution is None:
            self.initial_distribution = np.ones(self.n_states) / self.n_states
        else:
            initial_distribution = np.array(initial_distribution)
            if len(initial_distribution) != self.n_states:
                raise ValueError(f"Initial distribution must have {self.n_states} elements")
            self.initial_distribution = initial_distribution / initial_distribution.sum()
    
    def simulate(self, n_steps: int, initial_state: Optional[str] = None) -> List[str]:
        """
        Simulate a Markov chain path.
        
        Args:
            n_steps: Number of time steps to simulate
            initial_state: Starting state (if None, samples from initial_distribution)
            
        Returns:
            List of states visited at each time step
        """
        path = []
        
        # Determine initial state
        if initial_state is None:
            current_idx = np.random.choice(self.n_states, p=self.initial_distribution)
        else:
            current_idx = self.state_to_idx[initial_state]
        
        path.append(self.states[current_idx])
        
        # Simulate transitions
        for _ in range(n_steps - 1):
            # Sample next state from transition probabilities
            next_idx = np.random.choice(
                self.n_states,
                p=self.transition_matrix[current_idx, :]
            )
            path.append(self.states[next_idx])
            current_idx = next_idx
        
        return path
    
    def stationary_distribution(self, tolerance: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
        """
        Compute the stationary distribution of the Markov chain.
        
        Uses power iteration: π = π * P until convergence.
        
        Args:
            tolerance: Convergence tolerance
            max_iter: Maximum iterations
            
        Returns:
            Stationary probability distribution
        """
        pi = self.initial_distribution.copy()
        
        for _ in range(max_iter):
            pi_new = pi @ self.transition_matrix
            if np.linalg.norm(pi_new - pi) < tolerance:
                return pi_new
            pi = pi_new
        
        warnings.warn("Stationary distribution did not converge")
        return pi
    
    def apply_constraints(
        self,
        constraints: List[MarketConstraint],
        state_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Apply constraints to modify transition probabilities.
        
        Constraints affect transition probabilities by multiplying base probabilities
        by impact factors. The resulting matrix is renormalized.
        
        Args:
            constraints: List of MarketConstraint objects
            state_names: Optional list of state names (if different from self.states)
            
        Returns:
            Modified transition matrix accounting for constraints
        """
        if state_names is None:
            state_names = self.states
        
        modified_matrix = self.transition_matrix.copy()
        
        # Apply each constraint's impact
        for constraint in constraints:
            for i, from_state in enumerate(state_names):
                for j, to_state in enumerate(state_names):
                    impact = constraint.get_transition_impact(from_state, to_state)
                    modified_matrix[i, j] *= impact
        
        # Renormalize rows to ensure they sum to 1
        row_sums = modified_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        modified_matrix = modified_matrix / row_sums
        
        return modified_matrix
    
    def constrained_stationary_distribution(
        self,
        constraints: List[MarketConstraint],
        state_names: Optional[List[str]] = None,
        tolerance: float = 1e-6,
        max_iter: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute stationary distribution accounting for constraints.
        
        Args:
            constraints: List of MarketConstraint objects affecting transitions
            state_names: Optional list of state names
            tolerance: Convergence tolerance
            max_iter: Maximum iterations
            
        Returns:
            Tuple of (stationary distribution, modified transition matrix)
        """
        modified_matrix = self.apply_constraints(constraints, state_names)
        
        # Compute stationary distribution using modified matrix
        pi = self.initial_distribution.copy()
        
        for _ in range(max_iter):
            pi_new = pi @ modified_matrix
            if np.linalg.norm(pi_new - pi) < tolerance:
                return pi_new, modified_matrix
            pi = pi_new
        
        warnings.warn("Constrained stationary distribution did not converge")
        return pi, modified_matrix
    
    def n_step_transition(self, n: int) -> np.ndarray:
        """
        Compute n-step transition probabilities: P^n.
        
        Args:
            n: Number of steps
            
        Returns:
            n-step transition matrix
        """
        return np.linalg.matrix_power(self.transition_matrix, n)
    
    def expected_hitting_time(self, target_state: str, start_state: Optional[str] = None) -> float:
        """
        Compute expected time to hit a target state (first passage time).
        
        Args:
            target_state: State to hit
            start_state: Starting state (if None, uses initial distribution)
            
        Returns:
            Expected number of steps to reach target state
        """
        target_idx = self.state_to_idx[target_state]
        
        # Set up system of linear equations: E[T_i] = 1 + Σ P_ij * E[T_j]
        # For absorbing state: E[T_target] = 0
        A = np.eye(self.n_states) - self.transition_matrix
        A[target_idx, :] = 0
        A[target_idx, target_idx] = 1
        
        b = np.ones(self.n_states)
        b[target_idx] = 0
        
        expected_times = np.linalg.solve(A, b)
        
        if start_state is None:
            return np.dot(self.initial_distribution, expected_times)
        else:
            start_idx = self.state_to_idx[start_state]
            return expected_times[start_idx]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert transition matrix to DataFrame for easy inspection."""
        return pd.DataFrame(
            self.transition_matrix,
            index=self.states,
            columns=self.states
        )


def create_credit_rating_chain(
    default_prob: float = 0.02,
    upgrade_prob: float = 0.15,
    downgrade_prob: float = 0.20
) -> MarkovChain:
    """
    Create a standard credit rating Markov chain.
    
    States: AAA, AA, A, BBB, BB, B, CCC, D (Default)
    
    Args:
        default_prob: Base probability of defaulting from any state
        upgrade_prob: Probability of upgrading one notch
        downgrade_prob: Probability of downgrading one notch
        
    Returns:
        MarkovChain instance for credit ratings
    """
    states = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
    n = len(states)
    transition_matrix = np.zeros((n, n))
    
    # Default is absorbing state
    transition_matrix[n - 1, n - 1] = 1.0
    
    # For each non-default state
    for i in range(n - 1):
        # Stay in same state
        stay_prob = 1.0 - default_prob - upgrade_prob - downgrade_prob
        transition_matrix[i, i] = max(0, stay_prob)
        
        # Default
        transition_matrix[i, n - 1] = default_prob
        
        # Upgrade (can't upgrade from AAA)
        if i > 0:
            transition_matrix[i, i - 1] = upgrade_prob
        
        # Downgrade (can't downgrade from CCC)
        if i < n - 2:
            transition_matrix[i, i + 1] = downgrade_prob
    
    return MarkovChain(states, transition_matrix)


def create_risk_state_chain(
    low_to_medium: float = 0.1,
    medium_to_high: float = 0.15,
    high_to_default: float = 0.25,
    recovery_prob: float = 0.05
) -> MarkovChain:
    """
    Create a simple risk state Markov chain.
    
    States: Low, Medium, High, Default
    
    Args:
        low_to_medium: Transition probability from Low to Medium
        medium_to_high: Transition probability from Medium to High
        high_to_default: Transition probability from High to Default
        recovery_prob: Probability of recovering to a lower risk state
        
    Returns:
        MarkovChain instance for risk states
    """
    states = ["Low", "Medium", "High", "Default"]
    transition_matrix = np.array([
        [1 - low_to_medium, low_to_medium, 0, 0],
        [recovery_prob, 1 - recovery_prob - medium_to_high, medium_to_high, 0],
        [0, recovery_prob, 1 - recovery_prob - high_to_default, high_to_default],
        [0, 0, 0, 1.0]  # Default is absorbing
    ])
    
    return MarkovChain(states, transition_matrix)


def create_market_regime_chain(
    normal_stability: float = 0.85,
    stressed_stability: float = 0.60,
    crisis_stability: float = 0.40,
    recovery_stability: float = 0.70
) -> MarkovChain:
    """
    Create a market regime Markov chain.
    
    States: Normal, Stressed, Crisis, Recovery
    
    Args:
        normal_stability: Probability of staying in Normal state
        stressed_stability: Probability of staying in Stressed state
        crisis_stability: Probability of staying in Crisis state
        recovery_stability: Probability of staying in Recovery state
        
    Returns:
        MarkovChain instance for market regimes
    """
    states = ["Normal", "Stressed", "Crisis", "Recovery"]
    
    # Base transition probabilities
    # Normal can transition to Stressed
    normal_to_stressed = (1 - normal_stability) * 0.7
    normal_to_recovery = (1 - normal_stability) * 0.3
    
    # Stressed can transition to Normal, Crisis, or Recovery
    stressed_remaining = 1 - stressed_stability
    stressed_to_normal = stressed_remaining * 0.3
    stressed_to_crisis = stressed_remaining * 0.5
    stressed_to_recovery = stressed_remaining * 0.2
    
    # Crisis can transition to Recovery or stay
    crisis_remaining = 1 - crisis_stability
    crisis_to_recovery = crisis_remaining * 0.8
    crisis_to_stressed = crisis_remaining * 0.2
    
    # Recovery can transition to Normal or Stressed
    recovery_remaining = 1 - recovery_stability
    recovery_to_normal = recovery_remaining * 0.7
    recovery_to_stressed = recovery_remaining * 0.3
    
    transition_matrix = np.array([
        [normal_stability, normal_to_stressed, 0, normal_to_recovery],
        [stressed_to_normal, stressed_stability, stressed_to_crisis, stressed_to_recovery],
        [0, crisis_to_stressed, crisis_stability, crisis_to_recovery],
        [recovery_to_normal, recovery_to_stressed, 0, recovery_stability]
    ])
    
    return MarkovChain(states, transition_matrix)


class Varda:
    """
    Main Varda platform class for financial risk modeling and simulation.
    
    Varda models systemic risk, contagion, and credit scenarios using:
    - Network models to represent entity relationships
    - Fluid dynamics metaphors for risk propagation
    - Markov chain models for state transitions (credit ratings, risk states)
    - Monte Carlo simulations for scenario analysis
    - AI predictors for risk forecasting
    """
    
    def __init__(self, name: str = "Varda Risk Lab"):
        """
        Initialize the Varda platform.
        
        Args:
            name: Name identifier for this Varda instance
        """
        self.name = name
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.simulation_history: List[Dict[str, Any]] = []
        self.markov_chains: Dict[str, MarkovChain] = {}
        self.entity_states: Dict[str, str] = {}  # Track current state for each entity
        self.market_states: Dict[str, MarketState] = {}  # Market states as entities
        self.market_constraints: List[MarketConstraint] = []  # Global market constraints
        
    def add_entity(self, entity: Entity, initial_state: Optional[str] = None) -> None:
        """
        Add an entity to the network.
        
        Args:
            entity: Entity to add
            initial_state: Initial state for Markov chain modeling (optional)
            
        """
        self.entities[entity.id] = entity
        if initial_state is not None:
            self.entity_states[entity.id] = initial_state
        
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship between entities."""
        if relationship.source_id not in self.entities:
            raise ValueError(f"Source entity {relationship.source_id} not found")
        if relationship.target_id not in self.entities:
            raise ValueError(f"Target entity {relationship.target_id} not found")
        self.relationships.append(relationship)
        
    def get_network_adjacency(self) -> pd.DataFrame:
        """
        Build adjacency matrix representing the entity network.
        
        Returns:
            DataFrame with entities as rows/columns and relationship strengths as values
        """
        entity_ids = list(self.entities.keys())
        n = len(entity_ids)
        adj_matrix = np.zeros((n, n))
        
        id_to_idx = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}
        
        for rel in self.relationships:
            source_idx = id_to_idx[rel.source_id]
            target_idx = id_to_idx[rel.target_id]
            adj_matrix[source_idx, target_idx] = rel.strength
            
        return pd.DataFrame(adj_matrix, index=entity_ids, columns=entity_ids)
    
    def propagate_risk_fluid(
        self,
        initial_shock: Optional[Dict[str, float]] = None,
        diffusion_rate: float = 0.1,
        iterations: int = 10
    ) -> pd.DataFrame:
        """
        Simulate risk propagation using fluid dynamics-inspired diffusion model.
        
        Risk flows through the network like a fluid, with diffusion based on
        relationship strengths and connection topology.
        
        Args:
            initial_shock: Dict mapping entity_id to initial risk shock value
            diffusion_rate: Rate at which risk diffuses through connections (0-1)
            iterations: Number of propagation steps
            
        Returns:
            DataFrame with risk levels for each entity at each iteration
        """
        entity_ids = list(self.entities.keys())
        n = len(entity_ids)
        
        # Initialize risk levels
        risk_levels = np.zeros((iterations + 1, n))
        
        # Set initial conditions
        if initial_shock:
            for idx, entity_id in enumerate(entity_ids):
                risk_levels[0, idx] = initial_shock.get(entity_id, self.entities[entity_id].initial_risk_score)
        else:
            for idx, entity_id in enumerate(entity_ids):
                risk_levels[0, idx] = self.entities[entity_id].initial_risk_score
        
        # Get adjacency matrix
        adj_matrix = self.get_network_adjacency().values
        
        # Normalize adjacency matrix (row normalization for diffusion)
        row_sums = adj_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized_adj = adj_matrix / row_sums
        
        # Propagate risk through iterations
        for t in range(iterations):
            # Diffusion step: risk flows to neighbors
            diffused = normalized_adj @ risk_levels[t]
            # Update: blend current risk with diffused risk
            risk_levels[t + 1] = (1 - diffusion_rate) * risk_levels[t] + diffusion_rate * diffused
            # Ensure risk stays in [0, 1]
            risk_levels[t + 1] = np.clip(risk_levels[t + 1], 0, 1)
        
        # Convert to DataFrame
        columns = [f"iteration_{i}" for i in range(iterations + 1)]
        return pd.DataFrame(risk_levels.T, index=entity_ids, columns=columns)
    
    def monte_carlo_simulation(
        self,
        n_simulations: int = 1000,
        shock_distribution: str = "normal",
        shock_params: Optional[Dict[str, float]] = None,
        diffusion_rate: float = 0.1,
        iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations to assess risk under various scenarios.
        
        Args:
            n_simulations: Number of Monte Carlo runs
            shock_distribution: Distribution type for shocks ("normal", "uniform", "exponential")
            shock_params: Parameters for shock distribution
            diffusion_rate: Risk diffusion rate
            iterations: Propagation iterations per simulation
            
        Returns:
            Dictionary with simulation results including statistics and distributions
        """
        entity_ids = list(self.entities.keys())
        n_entities = len(entity_ids)
        
        # Default shock parameters
        if shock_params is None:
            shock_params = {"mean": 0.1, "std": 0.05} if shock_distribution == "normal" else {}
        
        # Storage for final risk levels across all simulations
        final_risks = np.zeros((n_simulations, n_entities))
        
        for sim in range(n_simulations):
            # Generate random initial shock
            initial_shock = {}
            for entity_id in entity_ids:
                if shock_distribution == "normal":
                    shock = np.random.normal(shock_params.get("mean", 0.1), 
                                            shock_params.get("std", 0.05))
                elif shock_distribution == "uniform":
                    shock = np.random.uniform(shock_params.get("low", 0.0),
                                             shock_params.get("high", 0.2))
                elif shock_distribution == "exponential":
                    shock = np.random.exponential(shock_params.get("scale", 0.1))
                else:
                    shock = 0.1
                
                initial_shock[entity_id] = np.clip(shock, 0, 1)
            
            # Run propagation
            risk_evolution = self.propagate_risk_fluid(
                initial_shock=initial_shock,
                diffusion_rate=diffusion_rate,
                iterations=iterations
            )
            
            # Store final risk levels
            final_risks[sim, :] = risk_evolution.iloc[:, -1].values
        
        # Compute statistics
        results = {
            "mean_risk": pd.Series(np.mean(final_risks, axis=0), index=entity_ids),
            "std_risk": pd.Series(np.std(final_risks, axis=0), index=entity_ids),
            "p5_risk": pd.Series(np.percentile(final_risks, 5, axis=0), index=entity_ids),
            "p95_risk": pd.Series(np.percentile(final_risks, 95, axis=0), index=entity_ids),
            "max_risk": pd.Series(np.max(final_risks, axis=0), index=entity_ids),
            "all_simulations": pd.DataFrame(final_risks, columns=entity_ids),
            "n_simulations": n_simulations
        }
        
        self.simulation_history.append(results)
        return results
    
    def identify_systemic_risk_hubs(self, threshold: float = 0.7) -> List[str]:
        """
        Identify entities that act as systemic risk hubs (highly connected, high risk).
        
        Args:
            threshold: Risk threshold for identifying hubs
            
        Returns:
            List of entity IDs that are systemic risk hubs
        """
        adj_matrix = self.get_network_adjacency()
        
        # Calculate connectivity (sum of incoming and outgoing connections)
        connectivity = adj_matrix.sum(axis=0) + adj_matrix.sum(axis=1)
        
        # Get current risk levels
        risk_levels = pd.Series({
            entity_id: entity.initial_risk_score 
            for entity_id, entity in self.entities.items()
        })
        
        # Identify hubs: high connectivity AND high risk
        hubs = []
        for entity_id in self.entities.keys():
            conn_score = connectivity[entity_id]
            risk_score = risk_levels[entity_id]
            hub_score = (conn_score / connectivity.max()) * risk_score if connectivity.max() > 0 else 0
            
            if hub_score >= threshold:
                hubs.append(entity_id)
        
        return hubs
    
    def get_risk_contagion_paths(
        self,
        source_entity_id: str,
        max_depth: int = 3
    ) -> List[List[str]]:
        """
        Find all paths through which risk can propagate from a source entity.
        
        Args:
            source_entity_id: Starting entity for contagion analysis
            max_depth: Maximum path length to explore
            
        Returns:
            List of paths (each path is a list of entity IDs)
        """
        if source_entity_id not in self.entities:
            raise ValueError(f"Entity {source_entity_id} not found")
        
        paths = []
        
        def dfs(current_id: str, path: List[str], depth: int):
            if depth >= max_depth:
                return
            
            for rel in self.relationships:
                if rel.source_id == current_id and rel.target_id not in path:
                    new_path = path + [rel.target_id]
                    paths.append(new_path)
                    dfs(rel.target_id, new_path, depth + 1)
        
        dfs(source_entity_id, [source_entity_id], 0)
        return paths
    
    def add_markov_chain(self, name: str, chain: MarkovChain) -> None:
        """Add a Markov chain model to Varda."""
        self.markov_chains[name] = chain
    
    def simulate_entity_transitions(
        self,
        chain_name: str,
        entity_ids: Optional[List[str]] = None,
        n_steps: int = 12,
        initial_states: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Simulate state transitions for entities using a Markov chain.
        
        Args:
            chain_name: Name of the Markov chain to use
            entity_ids: List of entity IDs to simulate (None = all entities)
            n_steps: Number of time steps to simulate
            initial_states: Dict mapping entity_id to initial state (overrides stored states)
            
        Returns:
            DataFrame with entity states at each time step
        """
        if chain_name not in self.markov_chains:
            raise ValueError(f"Markov chain '{chain_name}' not found")
        
        chain = self.markov_chains[chain_name]
        
        if entity_ids is None:
            entity_ids = list(self.entities.keys())
        
        # Determine initial states
        if initial_states is None:
            initial_states = {}
        
        # Simulate for each entity
        results = {}
        for entity_id in entity_ids:
            initial_state = initial_states.get(
                entity_id,
                self.entity_states.get(entity_id, None)
            )
            path = chain.simulate(n_steps, initial_state)
            results[entity_id] = path
            # Update stored state to final state
            self.entity_states[entity_id] = path[-1]
        
        # Convert to DataFrame
        return pd.DataFrame(results, index=[f"step_{i}" for i in range(n_steps)])
    
    def compute_default_probabilities(
        self,
        chain_name: str,
        horizon: int = 12,
        entity_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute default probabilities over time horizon using Markov chain.
        
        Args:
            chain_name: Name of the Markov chain to use
            horizon: Time horizon (number of steps)
            entity_ids: List of entity IDs (None = all entities)
            
        Returns:
            DataFrame with default probabilities for each entity at each step
        """
        if chain_name not in self.markov_chains:
            raise ValueError(f"Markov chain '{chain_name}' not found")
        
        chain = self.markov_chains[chain_name]
        
        # Find default state
        default_states = ["D", "Default", "default"]
        default_idx = None
        for state in default_states:
            if state in chain.states:
                default_idx = chain.state_to_idx[state]
                break
        
        if default_idx is None:
            raise ValueError(f"No default state found in chain '{chain_name}'")
        
        if entity_ids is None:
            entity_ids = list(self.entities.keys())
        
        # Compute n-step transition probabilities
        transition_n = chain.n_step_transition(horizon)
        
        # Get default probabilities for each entity
        default_probs = {}
        for entity_id in entity_ids:
            initial_state = self.entity_states.get(entity_id, None)
            if initial_state is None:
                # Use initial distribution
                prob = np.dot(chain.initial_distribution, transition_n[:, default_idx])
            else:
                initial_idx = chain.state_to_idx[initial_state]
                prob = transition_n[initial_idx, default_idx]
            default_probs[entity_id] = prob
        
        return pd.Series(default_probs, name=f"P(default|{horizon} steps)")
    
    def add_market_state(self, market_state: MarketState) -> None:
        """Add a market state to the analysis."""
        self.market_states[market_state.state_name] = market_state
        # Also add as an entity for network analysis
        entity = Entity(
            id=f"market_{market_state.state_name}",
            name=f"Market State: {market_state.state_name}",
            entity_type="market_state",
            initial_risk_score=1.0 - market_state.base_stability,
            metadata={
                "description": market_state.description,
                "economic_indicators": market_state.economic_indicators,
                **market_state.metadata
            }
        )
        self.entities[entity.id] = entity
    
    def add_market_constraint(self, constraint: MarketConstraint) -> None:
        """Add a market constraint that affects state transitions."""
        self.market_constraints.append(constraint)
    
    def analyze_market_steady_state(
        self,
        chain_name: str,
        constraints: Optional[List[MarketConstraint]] = None,
        return_modified_matrix: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze market state steady state probabilities accounting for constraints.
        
        This method treats each market state as an entity with constraints and
        computes the steady state distribution that reflects how constraints
        affect transition probabilities.
        
        Args:
            chain_name: Name of the Markov chain representing market states
            constraints: Optional list of constraints (uses global constraints if None)
            return_modified_matrix: If True, also return the modified transition matrix
            
        Returns:
            Dictionary with:
            - steady_state: Stationary probability distribution
            - state_names: List of state names
            - unconstrained_steady_state: Steady state without constraints
            - modified_transition_matrix: (if return_modified_matrix=True)
            - constraint_impacts: Summary of how constraints affected transitions
        """
        if chain_name not in self.markov_chains:
            raise ValueError(f"Markov chain '{chain_name}' not found")
        
        chain = self.markov_chains[chain_name]
        
        # Use provided constraints or global constraints
        if constraints is None:
            constraints = self.market_constraints
        
        # Get unconstrained steady state
        unconstrained_steady = chain.stationary_distribution()
        
        # Get constrained steady state
        if constraints:
            constrained_steady, modified_matrix = chain.constrained_stationary_distribution(
                constraints,
                state_names=chain.states
            )
        else:
            constrained_steady = unconstrained_steady
            modified_matrix = chain.transition_matrix
        
        # Analyze constraint impacts
        constraint_impacts = {}
        if constraints:
            for constraint in constraints:
                impacts = []
                for i, from_state in enumerate(chain.states):
                    for j, to_state in enumerate(chain.states):
                        impact = constraint.get_transition_impact(from_state, to_state)
                        if impact != 1.0:
                            impacts.append({
                                "transition": f"{from_state}->{to_state}",
                                "impact": impact,
                                "base_prob": chain.transition_matrix[i, j],
                                "modified_prob": modified_matrix[i, j]
                            })
                if impacts:
                    constraint_impacts[constraint.name] = impacts
        
        result = {
            "steady_state": pd.Series(constrained_steady, index=chain.states, name="Probability"),
            "state_names": chain.states,
            "unconstrained_steady_state": pd.Series(unconstrained_steady, index=chain.states),
            "constraint_impacts": constraint_impacts,
            "n_constraints": len(constraints) if constraints else 0
        }
        
        if return_modified_matrix:
            result["modified_transition_matrix"] = pd.DataFrame(
                modified_matrix,
                index=chain.states,
                columns=chain.states
            )
        
        return result
    
    def compare_market_scenarios(
        self,
        chain_name: str,
        scenario_constraints: Dict[str, List[MarketConstraint]]
    ) -> pd.DataFrame:
        """
        Compare steady state probabilities across different constraint scenarios.
        
        Args:
            chain_name: Name of the Markov chain
            scenario_constraints: Dict mapping scenario names to constraint lists
            
        Returns:
            DataFrame with steady state probabilities for each scenario
        """
        if chain_name not in self.markov_chains:
            raise ValueError(f"Markov chain '{chain_name}' not found")
        
        chain = self.markov_chains[chain_name]
        results = {}
        
        for scenario_name, constraints in scenario_constraints.items():
            analysis = self.analyze_market_steady_state(chain_name, constraints)
            results[scenario_name] = analysis["steady_state"]
        
        return pd.DataFrame(results)
    
    def summary(self) -> str:
        """Generate a summary of the Varda instance."""
        n_entities = len(self.entities)
        n_relationships = len(self.relationships)
        n_simulations = len(self.simulation_history)
        n_chains = len(self.markov_chains)
        n_market_states = len(self.market_states)
        n_constraints = len(self.market_constraints)
        
        entity_types = {}
        for entity in self.entities.values():
            entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1
        
        summary = f"""
Varda Risk Lab: {self.name}
================================
Entities: {n_entities}
  {', '.join(f'{k}: {v}' for k, v in entity_types.items())}
Relationships: {n_relationships}
Markov Chains: {n_chains}
Market States: {n_market_states}
Market Constraints: {n_constraints}
Simulations Run: {n_simulations}

Systemic Risk Hubs: {len(self.identify_systemic_risk_hubs())}
        """
        return summary.strip()


# Example usage and demonstration
if __name__ == "__main__":
    # Create a Varda instance
    varda = Varda("Example Risk Network")
    
    # Add entities (banks, borrowers, suppliers) with initial credit ratings
    varda.add_entity(Entity("bank1", "Bank A", "bank", initial_risk_score=0.2), initial_state="A")
    varda.add_entity(Entity("bank2", "Bank B", "bank", initial_risk_score=0.15), initial_state="AA")
    varda.add_entity(Entity("borrower1", "Corp X", "borrower", initial_risk_score=0.3), initial_state="BBB")
    varda.add_entity(Entity("borrower2", "Corp Y", "borrower", initial_risk_score=0.25), initial_state="BB")
    varda.add_entity(Entity("supplier1", "Supplier Z", "supplier", initial_risk_score=0.1), initial_state="A")
    
    # Add relationships
    varda.add_relationship(Relationship("bank1", "borrower1", "lending", strength=0.8, exposure=1000000))
    varda.add_relationship(Relationship("bank2", "borrower2", "lending", strength=0.7, exposure=500000))
    varda.add_relationship(Relationship("borrower1", "supplier1", "supply_chain", strength=0.6))
    varda.add_relationship(Relationship("borrower2", "supplier1", "supply_chain", strength=0.5))
    varda.add_relationship(Relationship("bank1", "bank2", "interbank", strength=0.4))
    
    # Print summary
    print(varda.summary())
    
    # Run risk propagation
    print("\nRunning risk propagation simulation...")
    risk_evolution = varda.propagate_risk_fluid(
        initial_shock={"borrower1": 0.5},
        diffusion_rate=0.15,
        iterations=10
    )
    print("\nRisk Evolution:")
    print(risk_evolution)
    
    # Identify systemic risk hubs
    print("\nSystemic Risk Hubs:")
    hubs = varda.identify_systemic_risk_hubs(threshold=0.3)
    for hub_id in hubs:
        entity = varda.entities[hub_id]
        print(f"  - {entity.name} ({hub_id}): risk={entity.initial_risk_score:.2f}")
    
    # Demonstrate Markov chain functionality
    print("\n" + "="*50)
    print("Markov Chain Example: Credit Rating Transitions")
    print("="*50)
    
    # Create a credit rating Markov chain
    credit_chain = create_credit_rating_chain(
        default_prob=0.02,
        upgrade_prob=0.15,
        downgrade_prob=0.20
    )
    varda.add_markov_chain("credit_ratings", credit_chain)
    
    print("\nTransition Matrix:")
    print(credit_chain.to_dataframe().round(3))
    
    # Simulate credit rating transitions
    print("\nSimulating credit rating transitions (12 months):")
    transitions = varda.simulate_entity_transitions(
        "credit_ratings",
        entity_ids=["bank1", "borrower1"],
        n_steps=12
    )
    print(transitions)
    
    # Compute default probabilities
    print("\nDefault Probabilities (12-month horizon):")
    default_probs = varda.compute_default_probabilities("credit_ratings", horizon=12)
    print(default_probs)
    
    # Stationary distribution
    print("\nStationary Distribution (long-run probabilities):")
    stationary = credit_chain.stationary_distribution()
    print(pd.Series(stationary, index=credit_chain.states).round(4))
    
    # Demonstrate market state analysis with constraints
    print("\n" + "="*50)
    print("Market State Analysis with Constraints")
    print("="*50)
    
    # Create market regime chain
    market_chain = create_market_regime_chain(
        normal_stability=0.85,
        stressed_stability=0.60,
        crisis_stability=0.40,
        recovery_stability=0.70
    )
    varda.add_markov_chain("market_regimes", market_chain)
    
    # Define market states as entities with constraints
    normal_state = MarketState(
        state_name="Normal",
        description="Stable market conditions with normal volatility",
        base_stability=0.85,
        economic_indicators={"inflation": 2.0, "unemployment": 4.0, "gdp_growth": 2.5}
    )
    
    stressed_state = MarketState(
        state_name="Stressed",
        description="Elevated risk and volatility",
        base_stability=0.60,
        economic_indicators={"inflation": 3.5, "unemployment": 6.0, "gdp_growth": 1.0}
    )
    
    crisis_state = MarketState(
        state_name="Crisis",
        description="Severe market disruption",
        base_stability=0.40,
        economic_indicators={"inflation": 5.0, "unemployment": 10.0, "gdp_growth": -2.0}
    )
    
    recovery_state = MarketState(
        state_name="Recovery",
        description="Post-crisis stabilization",
        base_stability=0.70,
        economic_indicators={"inflation": 2.5, "unemployment": 7.0, "gdp_growth": 1.5}
    )
    
    varda.add_market_state(normal_state)
    varda.add_market_state(stressed_state)
    varda.add_market_state(crisis_state)
    varda.add_market_state(recovery_state)
    
    # Define constraints that affect transitions
    high_inflation_constraint = MarketConstraint(
        name="High Inflation",
        constraint_type="economic",
        value=4.5,
        impact_on_transitions={
            "Normal->Stressed": 1.5,  # High inflation increases stress
            "Stressed->Crisis": 1.8,  # High inflation accelerates crisis
            "Recovery->Normal": 0.7,  # High inflation slows recovery
            "Normal->Normal": 0.9     # Less stability in normal state
        }
    )
    
    tight_policy_constraint = MarketConstraint(
        name="Tight Monetary Policy",
        constraint_type="policy",
        value=5.5,  # High interest rate
        impact_on_transitions={
            "Normal->Stressed": 1.3,
            "Stressed->Crisis": 1.2,
            "Crisis->Recovery": 0.8,  # Tight policy slows recovery
            "Recovery->Normal": 0.6
        }
    )
    
    liquidity_crisis_constraint = MarketConstraint(
        name="Liquidity Crisis",
        constraint_type="market",
        value=0.3,  # Low liquidity
        impact_on_transitions={
            "Normal->Stressed": 2.0,
            "Stressed->Crisis": 2.5,  # Liquidity crisis accelerates crisis
            "Crisis->Recovery": 0.5,  # Harder to recover
            "Stressed->Normal": 0.3   # Harder to return to normal
        }
    )
    
    varda.add_market_constraint(high_inflation_constraint)
    varda.add_market_constraint(tight_policy_constraint)
    varda.add_market_constraint(liquidity_crisis_constraint)
    
    # Analyze steady state with constraints
    print("\nBase Transition Matrix:")
    print(market_chain.to_dataframe().round(3))
    
    print("\nAnalyzing market steady state with constraints...")
    analysis = varda.analyze_market_steady_state(
        "market_regimes",
        return_modified_matrix=True
    )
    
    print("\nUnconstrained Steady State:")
    print(analysis["unconstrained_steady_state"].round(4))
    
    print("\nConstrained Steady State (accounting for constraints):")
    print(analysis["steady_state"].round(4))
    
    print("\nModified Transition Matrix (with constraints):")
    print(analysis["modified_transition_matrix"].round(3))
    
    print("\nConstraint Impacts:")
    for constraint_name, impacts in analysis["constraint_impacts"].items():
        print(f"\n  {constraint_name}:")
        for impact in impacts[:3]:  # Show first 3 impacts
            print(f"    {impact['transition']}: {impact['base_prob']:.3f} -> {impact['modified_prob']:.3f} (x{impact['impact']:.2f})")
    
    # Compare scenarios
    print("\n" + "="*50)
    print("Scenario Comparison")
    print("="*50)
    
    scenarios = {
        "Baseline": [],
        "High Inflation Only": [high_inflation_constraint],
        "Tight Policy Only": [tight_policy_constraint],
        "Liquidity Crisis Only": [liquidity_crisis_constraint],
        "All Constraints": [high_inflation_constraint, tight_policy_constraint, liquidity_crisis_constraint]
    }
    
    scenario_comparison = varda.compare_market_scenarios("market_regimes", scenarios)
    print("\nSteady State Probabilities by Scenario:")
    print(scenario_comparison.round(4))
