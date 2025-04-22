import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional

class Particle:
    """
    A single particle in the swarm.
    """
    def __init__(self, dim: int, bounds: List[Tuple[float, float]]):
        """
        Initialize a particle with random position and velocity.
        
        Args:
            dim: Dimensionality of the search space
            bounds: List of tuples (min, max) for each dimension
        """
        # Initialize position randomly within bounds
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        
        # Initialize velocity randomly
        velocity_bounds = [(-(high - low) / 10, (high - low) / 10) for low, high in bounds]
        self.velocity = np.array([np.random.uniform(low, high) for low, high in velocity_bounds])
        
        # Best position found so far (personal best)
        self.best_position = np.copy(self.position)
        
        # Best objective value found so far
        self.best_value = float('inf')  # For minimization problems
        
        # Current objective value
        self.current_value = float('inf')

    def update_velocity(self, global_best_position: np.ndarray, w: float, c1: float, c2: float):
        """
        Update the velocity of the particle.
        
        Args:
            global_best_position: Best position found by the swarm
            w: Inertia weight
            c1: Cognitive coefficient (personal best)
            c2: Social coefficient (global best)
        """
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        
        cognitive_component = c1 * r1 * (self.best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive_component + social_component

    def update_position(self, bounds: List[Tuple[float, float]]):
        """
        Update the position of the particle.
        
        Args:
            bounds: List of tuples (min, max) for each dimension
        """
        self.position = self.position + self.velocity
        
        # Enforce bounds
        for i, (low, high) in enumerate(bounds):
            if self.position[i] < low:
                self.position[i] = low
                self.velocity[i] *= -0.5  # Bounce back with reduced velocity
            elif self.position[i] > high:
                self.position[i] = high
                self.velocity[i] *= -0.5  # Bounce back with reduced velocity
                
    def evaluate(self, objective_func: Callable[[np.ndarray], float]):
        """
        Evaluate the objective function at the current position.
        
        Args:
            objective_func: The objective function to minimize
        """
        self.current_value = objective_func(self.position)
        
        # Update personal best if current position is better
        if self.current_value < self.best_value:
            self.best_value = self.current_value
            self.best_position = np.copy(self.position)
        
        return self.current_value


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization (PSO) algorithm for global optimization.
    """
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        num_particles: int = 30,
        max_iter: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        w_decay: float = 0.99,
        minimize: bool = True
    ):
        """
        Initialize the PSO optimizer.
        
        Args:
            objective_func: The objective function to optimize
            bounds: List of tuples (min, max) for each dimension
            num_particles: Number of particles in the swarm
            max_iter: Maximum number of iterations
            w: Inertia weight
            c1: Cognitive coefficient (personal best)
            c2: Social coefficient (global best)
            w_decay: Inertia weight decay factor per iteration
            minimize: If True, minimize the objective function; otherwise maximize
        """
        self.objective_func = objective_func if minimize else lambda x: -objective_func(x)
        self.bounds = bounds
        self.num_particles = num_particles
        self.dim = len(bounds)
        self.max_iter = max_iter
        self.w = w
        self.w_decay = w_decay
        self.c1 = c1
        self.c2 = c2
        
        # Initialize particles
        self.particles = [Particle(self.dim, self.bounds) for _ in range(num_particles)]
        
        # Global best position and value
        self.global_best_position = None
        self.global_best_value = float('inf')
        
        # Convergence history
        self.history = {
            'best_value': [],
            'avg_value': [],
            'diversity': []
        }
        
        # Initialize global best by evaluating all particles
        for particle in self.particles:
            particle.evaluate(self.objective_func)
            
            if particle.best_value < self.global_best_value:
                self.global_best_value = particle.best_value
                self.global_best_position = np.copy(particle.best_position)
                
    def compute_swarm_diversity(self) -> float:
        """
        Compute the diversity of the swarm (average distance from mean position).
        
        Returns:
            float: The diversity measure
        """
        mean_position = np.mean([p.position for p in self.particles], axis=0)
        distances = [np.linalg.norm(p.position - mean_position) for p in self.particles]
        return np.mean(distances)
                
    def optimize(self, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Run the PSO optimization process.
        
        Args:
            verbose: If True, print progress information
        
        Returns:
            Tuple of (best_position, best_value)
        """
        current_w = self.w
        
        for iteration in range(self.max_iter):
            # Update each particle
            for particle in self.particles:
                # Update velocity and position
                particle.update_velocity(self.global_best_position, current_w, self.c1, self.c2)
                particle.update_position(self.bounds)
                
                # Evaluate new position
                particle.evaluate(self.objective_func)
                
                # Update global best if needed
                if particle.best_value < self.global_best_value:
                    self.global_best_value = particle.best_value
                    self.global_best_position = np.copy(particle.best_position)
            
            # Calculate average value for this iteration
            avg_value = np.mean([p.current_value for p in self.particles])
            
            # Calculate diversity for this iteration
            diversity = self.compute_swarm_diversity()
            
            # Store history
            self.history['best_value'].append(self.global_best_value)
            self.history['avg_value'].append(avg_value)
            self.history['diversity'].append(diversity)
            
            # Decay inertia weight
            current_w *= self.w_decay
            
            if verbose and (iteration % (self.max_iter // 10) == 0 or iteration == self.max_iter - 1):
                print(f"Iteration {iteration}: Best Value = {self.global_best_value}, Avg Value = {avg_value}, Diversity = {diversity:.6f}")
        
        return self.global_best_position, self.global_best_value
    
    def plot_convergence(self):
        """
        Plot the convergence history of the optimization.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot best and average values
        iterations = range(len(self.history['best_value']))
        ax1.plot(iterations, self.history['best_value'], 'b-', label='Best Value')
        ax1.plot(iterations, self.history['avg_value'], 'r--', label='Average Value')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('PSO Convergence')
        ax1.legend()
        ax1.grid(True)
        
        # Plot diversity
        ax2.plot(iterations, self.history['diversity'], 'g-')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Swarm Diversity')
        ax2.set_title('Swarm Diversity over Time')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def visualize_2d(self, resolution=100, show_particles=True):
        """
        Visualize the optimization landscape and particles for 2D problems.
        Only works for 2D optimization problems.
        
        Args:
            resolution: Grid resolution for contour plot
            show_particles: If True, show the particles on the plot
        """
        if self.dim != 2:
            print(f"This method only works for 2D problems. Current dimension: {self.dim}")
            return
            
        # Create a grid of points
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], resolution)
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate the objective function on the grid
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = self.objective_func(np.array([X[i, j], Y[i, j]]))
                
        # Create the plot
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.8)
        plt.colorbar(contour, label='Objective Value')
        
        # Plot particles if requested
        if show_particles:
            positions = np.array([p.position for p in self.particles])
            plt.scatter(positions[:, 0], positions[:, 1], c='red', marker='o', 
                        label='Particles', alpha=0.7)
            
            # Plot global best
            plt.scatter([self.global_best_position[0]], [self.global_best_position[1]], 
                        c='yellow', marker='*', s=200, label='Global Best', edgecolors='black')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Optimization Landscape with Particles')
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage with common benchmark functions
def rastrigin(x):
    """
    Rastrigin function - a non-convex function with many local minima.
    Global minimum at x = 0 with f(x) = 0.
    """
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def rosenbrock(x):
    """
    Rosenbrock function - a non-convex function with a narrow valley.
    Global minimum at x = [1, 1, ..., 1] with f(x) = 0.
    """
    return sum([100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1)])

def sphere(x):
    """
    Sphere function - a simple convex function.
    Global minimum at x = 0 with f(x) = 0.
    """
    return sum([xi**2 for xi in x])

def ackley(x):
    """
    Ackley function - a non-convex function with many local minima.
    Global minimum at x = 0 with f(x) = 0.
    """
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x)
    sum1 = -a * np.exp(-b * np.sqrt(sum([xi**2 for xi in x]) / n))
    sum2 = -np.exp(sum([np.cos(c * xi) for xi in x]) / n)
    return sum1 + sum2 + a + np.exp(1)


# Example: Optimize the Rastrigin function in 2D
if __name__ == "__main__":
    # Problem setup
    dim = 2
    bounds = [(-5.12, 5.12)] * dim  # Rastrigin is typically evaluated in [-5.12, 5.12]
    
    # Create PSO optimizer
    pso = ParticleSwarmOptimizer(
        objective_func=rastrigin,
        bounds=bounds,
        num_particles=50,
        max_iter=100,
        w=0.7,
        c1=1.5,
        c2=1.5,
        w_decay=0.99
    )
    
    # Run optimization
    best_position, best_value = pso.optimize(verbose=True)
    
    print("\nOptimization Results:")
    print(f"Best Position: {best_position}")
    print(f"Best Value: {best_value}")
    
    # Visualize results
    pso.plot_convergence()
    pso.visualize_2d()