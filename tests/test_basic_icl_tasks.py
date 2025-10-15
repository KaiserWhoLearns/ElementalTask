"""
Test basic ICL tasks for function vector basis formation and PCA analysis.
"""
import pytest
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

from tasks.base_task import BaseTask, TaskConfig
from function_vecs.extract_function_vecs import (
    ExtractConfig, 
    extract_task_function_vec,
    stack_function_vecs,
    build_skill_basis,
    Headset,
    _batch_per_head_contribs,
    get_shuffled_prompts,
    _sample_task_prompts
)



class SimpleArithmeticTask(BaseTask):
    """Basic addition task: a + b = c"""
    
    def __init__(self):
        config = TaskConfig(
            name="simple_arithmetic",
            input_column="input",
            output_column="output",
            description="Solve the arithmetic problem."
        )
        super().__init__(config)
        
    def get_split(self, split_name: str):
        # Generate simple addition problems
        data = []
        for a in range(1, 6):
            for b in range(1, 6):
                c = a + b
                data.append({
                    "input": f"{a} + {b}",
                    "output": str(c)
                })
        return data[:20]  # Keep it small
    
    def build_prompt(self, row):
        # ICL format with 2 demonstrations
        prompt = "3 + 2 -> 5\n4 + 1 -> 5\n"
        prompt += f"{row['input']} ->"
        return prompt
    
    def evaluate(self, predictions, split="test", **kwargs):
        """Simple exact match evaluation."""
        ground_truth = self.get_ground_truth(split)
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if pred.strip() == gt.strip())
        return {"accuracy": correct / len(ground_truth)}


class SimpleNegationTask(BaseTask):
    """Basic negation task: not X = Y"""
    
    def __init__(self):
        config = TaskConfig(
            name="simple_negation", 
            input_column="input",
            output_column="output",
            description="Negate the input."
        )
        super().__init__(config)
        
    def get_split(self, split_name: str):
        # Simple yes/no negation
        data = [
            {"input": "yes", "output": "no"},
            {"input": "no", "output": "yes"},
            {"input": "true", "output": "false"},
            {"input": "false", "output": "true"},
            {"input": "on", "output": "off"},
            {"input": "off", "output": "on"},
            {"input": "up", "output": "down"},
            {"input": "down", "output": "up"},
            {"input": "left", "output": "right"},
            {"input": "right", "output": "left"},
        ]
        return data
    
    def build_prompt(self, row):
        # ICL format
        prompt = "yes -> no\ntrue -> false\n"
        prompt += f"{row['input']} ->"
        return prompt
    
    def evaluate(self, predictions, split="test", **kwargs):
        """Simple exact match evaluation."""
        ground_truth = self.get_ground_truth(split)
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if pred.strip() == gt.strip())
        return {"accuracy": correct / len(ground_truth)}


class SimpleCapitalizationTask(BaseTask):
    """Basic capitalization task: lowercase -> UPPERCASE"""
    
    def __init__(self):
        config = TaskConfig(
            name="simple_capitalization",
            input_column="input", 
            output_column="output",
            description="Convert to uppercase."
        )
        super().__init__(config)
        
    def get_split(self, split_name: str):
        words = ["cat", "dog", "sun", "moon", "tree", "bird", "fish", "star", "rock", "wave"]
        data = []
        for word in words:
            data.append({
                "input": word,
                "output": word.upper()
            })
        return data
    
    def build_prompt(self, row):
        prompt = "cat -> CAT\ndog -> DOG\n"
        prompt += f"{row['input']} ->"
        return prompt
    
    def evaluate(self, predictions, split="test", **kwargs):
        """Simple exact match evaluation."""
        ground_truth = self.get_ground_truth(split)
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if pred.strip() == gt.strip())
        return {"accuracy": correct / len(ground_truth)}


class SimpleRhymingTask(BaseTask):
    """Basic rhyming task: cat -> bat, dog -> log"""
    
    def __init__(self):
        config = TaskConfig(
            name="simple_rhyming",
            input_column="input",
            output_column="output", 
            description="Find a rhyming word."
        )
        super().__init__(config)
        
    def get_split(self, split_name: str):
        rhymes = [
            ("cat", "bat"),
            ("dog", "log"),
            ("sun", "fun"),
            ("tree", "bee"),
            ("car", "star"),
            ("hat", "rat"),
            ("big", "pig"),
            ("run", "gun"),
            ("sea", "tea"),
            ("red", "bed")
        ]
        data = []
        for word, rhyme in rhymes:
            data.append({
                "input": word,
                "output": rhyme
            })
        return data
    
    def build_prompt(self, row):
        prompt = "cat -> bat\ndog -> log\n"
        prompt += f"{row['input']} ->"
        return prompt
    
    def evaluate(self, predictions, split="test", **kwargs):
        """Simple exact match evaluation."""
        ground_truth = self.get_ground_truth(split)
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if pred.strip() == gt.strip())
        return {"accuracy": correct / len(ground_truth)}


class FirstCharacterTask(BaseTask):
    """Extract first character: hello -> h"""
    
    def __init__(self):
        config = TaskConfig(
            name="first_character",
            input_column="input",
            output_column="output",
            description="Extract the first character."
        )
        super().__init__(config)
        
    def get_split(self, split_name: str):
        words = ["apple", "banana", "cherry", "dragon", "elephant", "forest", "garden", "house", "island", "jungle"]
        data = []
        for word in words:
            data.append({
                "input": word,
                "output": word[0]
            })
        return data
    
    def build_prompt(self, row):
        prompt = "apple -> a\nbanana -> b\n"
        prompt += f"{row['input']} ->"
        return prompt
    
    def evaluate(self, predictions, split="test", **kwargs):
        ground_truth = self.get_ground_truth(split)
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if pred.strip() == gt.strip())
        return {"accuracy": correct / len(ground_truth)}


class LastCharacterTask(BaseTask):
    """Extract last character: hello -> o"""
    
    def __init__(self):
        config = TaskConfig(
            name="last_character",
            input_column="input",
            output_column="output",
            description="Extract the last character."
        )
        super().__init__(config)
        
    def get_split(self, split_name: str):
        words = ["cat", "dog", "sun", "moon", "tree", "bird", "fish", "star", "rock", "wave"]
        data = []
        for word in words:
            data.append({
                "input": word,
                "output": word[-1]
            })
        return data
    
    def build_prompt(self, row):
        prompt = "cat -> t\ndog -> g\n"
        prompt += f"{row['input']} ->"
        return prompt
    
    def evaluate(self, predictions, split="test", **kwargs):
        ground_truth = self.get_ground_truth(split)
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if pred.strip() == gt.strip())
        return {"accuracy": correct / len(ground_truth)}


class ReverseStringTask(BaseTask):
    """Reverse string: cat -> tac"""
    
    def __init__(self):
        config = TaskConfig(
            name="reverse_string",
            input_column="input",
            output_column="output",
            description="Reverse the string."
        )
        super().__init__(config)
        
    def get_split(self, split_name: str):
        words = ["cat", "dog", "sun", "car", "hat", "big", "run", "sea", "red", "fox"]
        data = []
        for word in words:
            data.append({
                "input": word,
                "output": word[::-1]
            })
        return data
    
    def build_prompt(self, row):
        prompt = "cat -> tac\ndog -> god\n"
        prompt += f"{row['input']} ->"
        return prompt
    
    def evaluate(self, predictions, split="test", **kwargs):
        ground_truth = self.get_ground_truth(split)
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if pred.strip() == gt.strip())
        return {"accuracy": correct / len(ground_truth)}


class VowelCountTask(BaseTask):
    """Count vowels: hello -> 2"""
    
    def __init__(self):
        config = TaskConfig(
            name="vowel_count",
            input_column="input",
            output_column="output",
            description="Count the number of vowels."
        )
        super().__init__(config)
        
    def get_split(self, split_name: str):
        words = ["cat", "hello", "tree", "banana", "orange", "apple", "house", "elephant", "ocean", "umbrella"]
        data = []
        for word in words:
            vowel_count = sum(1 for char in word.lower() if char in 'aeiou')
            data.append({
                "input": word,
                "output": str(vowel_count)
            })
        return data
    
    def build_prompt(self, row):
        prompt = "cat -> 1\nhello -> 2\n"
        prompt += f"{row['input']} ->"
        return prompt
    
    def evaluate(self, predictions, split="test", **kwargs):
        ground_truth = self.get_ground_truth(split)
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if pred.strip() == gt.strip())
        return {"accuracy": correct / len(ground_truth)}


class StringLengthTask(BaseTask):
    """String length: cat -> 3"""
    
    def __init__(self):
        config = TaskConfig(
            name="string_length",
            input_column="input",
            output_column="output",
            description="Count the string length."
        )
        super().__init__(config)
        
    def get_split(self, split_name: str):
        words = ["cat", "dog", "hello", "world", "a", "hi", "python", "task", "simple", "test"]
        data = []
        for word in words:
            data.append({
                "input": word,
                "output": str(len(word))
            })
        return data
    
    def build_prompt(self, row):
        prompt = "cat -> 3\nhello -> 5\n"
        prompt += f"{row['input']} ->"
        return prompt
    
    def evaluate(self, predictions, split="test", **kwargs):
        ground_truth = self.get_ground_truth(split)
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if pred.strip() == gt.strip())
        return {"accuracy": correct / len(ground_truth)}


class AddOneTask(BaseTask):
    """Add one to number: 5 -> 6"""
    
    def __init__(self):
        config = TaskConfig(
            name="add_one",
            input_column="input",
            output_column="output",
            description="Add one to the number."
        )
        super().__init__(config)
        
    def get_split(self, split_name: str):
        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data = []
        for num in numbers:
            data.append({
                "input": str(num),
                "output": str(num + 1)
            })
        return data
    
    def build_prompt(self, row):
        prompt = "3 -> 4\n7 -> 8\n"
        prompt += f"{row['input']} ->"
        return prompt
    
    def evaluate(self, predictions, split="test", **kwargs):
        ground_truth = self.get_ground_truth(split)
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if pred.strip() == gt.strip())
        return {"accuracy": correct / len(ground_truth)}


@pytest.mark.integration
def test_basic_function_vector_extraction():
    """Test extracting function vectors from basic ICL tasks."""
    
    # Use tiny model for speed
    config = ExtractConfig(
        model_name="distilgpt2",
        device="cpu",  # Force CPU to avoid CUDA issues
        batch_size=4,
        num_samples_per_task=8,
        layers=[5],  # Just test one layer
        topk_heads=4
    )
    
    # Load model once
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_name).eval()
    
    # Create simple headset (just pick a few heads)
    headset = Headset(mode="topk", heads=[(5, 0), (5, 1), (5, 2), (5, 3)])
    
    # Test tasks
    tasks = [
        SimpleArithmeticTask(),
        SimpleNegationTask(), 
        SimpleCapitalizationTask(),
        SimpleRhymingTask(),
        FirstCharacterTask(),
        LastCharacterTask(),
        ReverseStringTask(),
        VowelCountTask(),
        StringLengthTask(),
        AddOneTask()
    ]
    
    # Extract function vectors
    function_vecs = []
    for task in tasks:
        print(f"\nTesting task: {task.config.name}")
        
        # Test basic prompt sampling
        prompts = _sample_task_prompts(task, 5)
        print(f"Sample prompts ({len(prompts)}):")
        for i, p in enumerate(prompts[:2]):
            print(f"  {i+1}: {repr(p)}")
            
        # Test control generation
        controls = get_shuffled_prompts(task, 3)
        print(f"Control prompts ({len(controls)}):")
        for i, c in enumerate(controls[:2]):
            print(f"  {i+1}: {repr(c)}")
        
        # Test contribution extraction
        contribs = _batch_per_head_contribs(
            model, tokenizer, prompts[:2], layer_idx=5, device="cpu"
        )
        print(f"Contributions shape: {contribs.shape}")
        print(f"Contribution stats - mean: {contribs.mean():.4f}, std: {contribs.std():.4f}")
        
        # Extract function vector
        func_vec = extract_task_function_vec(task, config, headset, model, tokenizer)
        function_vecs.append(func_vec)
        
        print(f"Function vector shape: {func_vec.function_vec.shape}")
        print(f"Function vector norm: {np.linalg.norm(func_vec.function_vec):.4f}")
        print(f"Function vector preview: {func_vec.function_vec[:5]}")
    
    assert len(function_vecs) == 10
    for fv in function_vecs:
        assert fv.function_vec.shape[0] > 0
        # Should be L2 normalized
        norm = np.linalg.norm(fv.function_vec)
        assert abs(norm - 1.0) < 1e-6, f"Expected L2 norm=1, got {norm}"


@pytest.mark.integration  
def test_basis_formation_and_pca():
    """Test stacking function vectors and building skill basis with PCA."""
    
    config = ExtractConfig(
        model_name="distilgpt2", 
        device="cpu",
        batch_size=4,
        num_samples_per_task=6,
        layers=[5],
        topk_heads=4
    )
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_name).eval()
    
    # Simple headset
    headset = Headset(mode="topk", heads=[(5, 0), (5, 1), (5, 2), (5, 3)])
    
    # Create tasks
    tasks = [
        SimpleArithmeticTask(),
        SimpleNegationTask(),
        SimpleCapitalizationTask(), 
        SimpleRhymingTask(),
        FirstCharacterTask(),
        LastCharacterTask(),
        ReverseStringTask(),
        VowelCountTask(),
        StringLengthTask(),
        AddOneTask()
    ]
    
    # Extract function vectors
    print("\n=== Extracting Function Vectors ===")
    function_vecs = []
    for task in tasks:
        func_vec = extract_task_function_vec(task, config, headset, model, tokenizer)
        function_vecs.append(func_vec)
        print(f"{task.config.name}: shape={func_vec.function_vec.shape}, norm={np.linalg.norm(func_vec.function_vec):.4f}")
    
    # Stack into matrix
    print("\n=== Stacking Function Vectors ===")
    task_matrix = stack_function_vecs(function_vecs)
    print(f"Task matrix shape: {task_matrix.V.shape}")
    print(f"Task names: {task_matrix.task_names}")
    
    # Build skill basis with SVD
    print("\n=== Building Skill Basis (SVD) ===")
    skill_basis = build_skill_basis(task_matrix, method="svd", k=6)  # More components for richer analysis
    print(f"U shape: {skill_basis.U.shape}")
    print(f"S shape: {skill_basis.S.shape}")  
    print(f"Vt shape: {skill_basis.Vt.shape}")
    print(f"Singular values: {skill_basis.S}")
    
    # Check SVD properties
    assert skill_basis.U.shape[1] == 6  # k=6 components
    assert skill_basis.S.shape[0] == 6
    assert skill_basis.Vt.shape[0] == 6
    assert skill_basis.Vt.shape[1] == len(tasks)
    
    # Singular values should be sorted in descending order
    assert np.all(skill_basis.S[:-1] >= skill_basis.S[1:])
    
    print("\n=== PCA Analysis ===")
    # Compute explained variance ratios
    total_var = np.sum(skill_basis.S ** 2)
    explained_var_ratio = (skill_basis.S ** 2) / total_var
    print(f"Explained variance ratios: {explained_var_ratio}")
    print(f"Cumulative explained variance: {np.cumsum(explained_var_ratio)}")
    
    # Project tasks onto principal components  
    print(f"\nTask projections onto first 6 components:")
    projections = skill_basis.Vt.T  # (n_tasks, n_components)
    for i, task_name in enumerate(task_matrix.task_names):
        proj = projections[i, :]
        print(f"  {task_name:20s}: [{proj[0]:6.3f}, {proj[1]:6.3f}, {proj[2]:6.3f}, {proj[3]:6.3f}, {proj[4]:6.3f}, {proj[5]:6.3f}]")
    
    # Basic sanity checks
    assert explained_var_ratio[0] > 0.1  # First component should explain some variance
    assert np.sum(explained_var_ratio) <= 1.0  # Can't explain more than 100%
    
    print("\n=== Function Vector Similarities ===")
    # Compute pairwise similarities between function vectors
    V = task_matrix.V  # (d_model, n_tasks)
    similarities = V.T @ V  # (n_tasks, n_tasks) - cosine similarities since vectors are normalized
    
    for i, name_i in enumerate(task_matrix.task_names):
        for j, name_j in enumerate(task_matrix.task_names):
            if i < j:  # Only upper triangle
                sim = similarities[i, j]
                print(f"  {name_i} <-> {name_j}: {sim:.3f}")

    # ====== VISUALIZATION SECTION - REMOVE BEFORE FINAL ======
    print("\n=== 📊 PCA Visualization (REMOVE THIS SECTION) ===")
    try:
        import matplotlib.pyplot as plt
        
        # Create 2D and 3D plots of task projections
        projections = skill_basis.Vt.T  # (n_tasks, n_components)
        
        # 2D plot: PC1 vs PC2
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(projections[:, 0], projections[:, 1], s=100, alpha=0.7)
        for i, name in enumerate(task_matrix.task_names):
            plt.annotate(name, (projections[i, 0], projections[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel(f'PC1 ({explained_var_ratio[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({explained_var_ratio[1]:.1%} variance)')
        plt.title('Task Projections: PC1 vs PC2')
        plt.grid(True, alpha=0.3)
        
        # 2D plot: PC1 vs PC3  
        plt.subplot(1, 2, 2)
        plt.scatter(projections[:, 0], projections[:, 2], s=100, alpha=0.7, color='orange')
        for i, name in enumerate(task_matrix.task_names):
            plt.annotate(name, (projections[i, 0], projections[i, 2]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel(f'PC1 ({explained_var_ratio[0]:.1%} variance)')
        plt.ylabel(f'PC3 ({explained_var_ratio[2]:.1%} variance)')
        plt.title('Task Projections: PC1 vs PC3')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/projects/bfcu/ElementalTask/task_pca_2d.png', dpi=150, bbox_inches='tight')
        print("📈 Saved 2D PCA plot to: task_pca_2d.png")
        
        # 3D plot: PC1, PC2, PC3
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color tasks by type for better visualization
        task_colors = {
            'simple_arithmetic': 'red',
            'add_one': 'red', 
            'string_length': 'blue',
            'vowel_count': 'blue',
            'first_character': 'green',
            'last_character': 'green',
            'simple_capitalization': 'orange',
            'simple_rhyming': 'orange',
            'reverse_string': 'orange',
            'simple_negation': 'purple'
        }
        
        colors = [task_colors.get(name, 'gray') for name in task_matrix.task_names]
        
        scatter = ax.scatter(projections[:, 0], projections[:, 1], projections[:, 2], 
                           c=colors, s=100, alpha=0.7)
        
        for i, name in enumerate(task_matrix.task_names):
            ax.text(projections[i, 0], projections[i, 1], projections[i, 2], 
                   name, fontsize=8)
        
        ax.set_xlabel(f'PC1 ({explained_var_ratio[0]:.1%})')
        ax.set_ylabel(f'PC2 ({explained_var_ratio[1]:.1%})')
        ax.set_zlabel(f'PC3 ({explained_var_ratio[2]:.1%})')
        ax.set_title('Task Projections in 3D PCA Space')
        
        plt.savefig('/projects/bfcu/ElementalTask/task_pca_3d.png', dpi=150, bbox_inches='tight')
        print("📈 Saved 3D PCA plot to: task_pca_3d.png")
        
        # Task type clustering analysis
        print("\n=== Task Type Analysis ===")
        numerical_tasks = ['simple_arithmetic', 'add_one', 'string_length', 'vowel_count']
        character_tasks = ['first_character', 'last_character']
        transformation_tasks = ['simple_capitalization', 'simple_rhyming', 'reverse_string']
        logic_tasks = ['simple_negation']
        
        def get_centroid(task_list):
            indices = [task_matrix.task_names.index(name) for name in task_list if name in task_matrix.task_names]
            return projections[indices].mean(axis=0)
        
        numerical_centroid = get_centroid(numerical_tasks)
        character_centroid = get_centroid(character_tasks)
        transformation_centroid = get_centroid(transformation_tasks)
        
        print(f"Numerical tasks centroid:     [{numerical_centroid[0]:6.3f}, {numerical_centroid[1]:6.3f}, {numerical_centroid[2]:6.3f}]")
        print(f"Character tasks centroid:     [{character_centroid[0]:6.3f}, {character_centroid[1]:6.3f}, {character_centroid[2]:6.3f}]")  
        print(f"Transformation tasks centroid:[{transformation_centroid[0]:6.3f}, {transformation_centroid[1]:6.3f}, {transformation_centroid[2]:6.3f}]")
        
        # Distance between centroids
        dist_num_char = np.linalg.norm(numerical_centroid - character_centroid)
        dist_num_trans = np.linalg.norm(numerical_centroid - transformation_centroid)
        dist_char_trans = np.linalg.norm(character_centroid - transformation_centroid)
        
        print(f"\nCentroid distances:")
        print(f"  Numerical ↔ Character:      {dist_num_char:.3f}")
        print(f"  Numerical ↔ Transformation: {dist_num_trans:.3f}")
        print(f"  Character ↔ Transformation: {dist_char_trans:.3f}")
        
        plt.close('all')  # Clean up plots
        
    except ImportError:
        print("❌ matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"❌ Visualization error: {e}")
    # ============== END VISUALIZATION SECTION ================


if __name__ == "__main__":
    # Run basic tests
    test_basic_function_vector_extraction()
    test_basis_formation_and_pca()
    print("\nAll tests passed!")
