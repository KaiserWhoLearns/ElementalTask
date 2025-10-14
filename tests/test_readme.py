# Test the README example
from function_vecs.extract_function_vecs import extract_function_vector_simple, extract_task_function_vec, ExtractConfig, Headset
from tasks.base_task import TaskConfig
from tasks.registry import get_task

print("Testing README example...")

# NEW: Simplified one-stop interface
print("\n=== Testing Simplified Interface ===")
function_vec_simple = extract_function_vector_simple("simple_icl", num_samples=3)
print(f"Simple interface - Function vector shape: {function_vec_simple.function_vec.shape}")
print(f"Task name: {function_vec_simple.task_name}")

# ORIGINAL: More detailed interface for advanced usage
print("\n=== Testing Original Detailed Interface ===")

# 1. Define your task or import one

# if you define a new task from in-mem examples
task_config = TaskConfig(
    name="uppercase_conversion",
    data_format="memory", 
    in_memory_data=[
        {"input": "a", "output": "A"},
        {"input": "b", "output": "B"},
        {"input": "c", "output": "C"}
    ]
)
# Create task from config
task_from_config = get_task("simple_icl", task_config)  # Use simple_icl as base class

# use a preexisting task
simple_icl_config = TaskConfig(
    name="simple_icl", 
    data_path="dataset/simple.csv",
    input_column="question",
    output_column="answer"
)
task = get_task("simple_icl", simple_icl_config)

# 2. Configure extraction
extract_config = ExtractConfig(
    model_name="gpt2",
    num_samples_per_task=3,
    topk_heads=3
)

# 3. Create headset (will be auto-determined if not provided)
head_set = Headset(mode="topk", heads=[(0, 0), (0, 1), (0, 2)])

# 4. Extract function vector
print("Extracting function vector from task_config...")
function_vec_from_config = extract_task_function_vec(task_from_config, extract_config, head_set)
print(function_vec_from_config)

print("Extracting function vector from existing task...")
function_vec_from_task = extract_task_function_vec(task, extract_config, head_set)
print(function_vec_from_task)