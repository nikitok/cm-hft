//! Strategy registry and factory.
//!
//! [`StrategyRegistry`] maps strategy names to factory functions, allowing
//! strategies to be instantiated by name at runtime. This enables
//! configuration-driven strategy selection without dynamic loading.

use std::collections::HashMap;

use super::strategies;
use super::traits::{Strategy, StrategyParams};

/// Registry of available strategies.
///
/// Maps strategy names to factory functions that create boxed trait objects.
pub struct StrategyRegistry {
    factories: HashMap<String, Box<dyn Fn(&StrategyParams) -> Box<dyn Strategy>>>,
}

impl StrategyRegistry {
    /// Create a new, empty registry.
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register a strategy factory under the given name.
    pub fn register<F>(&mut self, name: &str, factory: F)
    where
        F: Fn(&StrategyParams) -> Box<dyn Strategy> + 'static,
    {
        self.factories.insert(name.to_string(), Box::new(factory));
    }

    /// Create a strategy instance by name. Returns `None` if the name is not registered.
    pub fn create(&self, name: &str, params: &StrategyParams) -> Option<Box<dyn Strategy>> {
        self.factories.get(name).map(|f| f(params))
    }

    /// List all registered strategy names.
    pub fn available_strategies(&self) -> Vec<String> {
        let mut names: Vec<String> = self.factories.keys().cloned().collect();
        names.sort();
        names
    }
}

impl Default for StrategyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a default registry with all built-in strategies.
pub fn default_registry() -> StrategyRegistry {
    let mut registry = StrategyRegistry::new();
    registry.register("market_making", |params| {
        Box::new(strategies::MarketMakingStrategy::from_params(params))
    });
    // Alias for convenience / config compatibility
    registry.register("simple_mm", |params| {
        Box::new(strategies::MarketMakingStrategy::from_params(params))
    });
    registry
}
