import json
import traceback
import numpy as np
import time

nb_path = 'neural-network.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Execution environment for notebook code
env = {'np': np, 'time': time}

# Execute all code cells except the main training invocation and image recognition
# (ensure function definitions like `learn` are executed)
skip_keywords = [
    "training_data, validation_data, test_data = load_data_wrapper()",
    'nn = init_network([784, 30, 10])',
    "for file in range(0,10):",
    "recognize_image('",
    'print(\'Learning process started...\')',
    'learn(nn, training_data, epochs, mini_batch_size, learning_rate, test_data)'
]

for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    source = ''.join(cell.get('source', []))
    if any(k in source for k in skip_keywords):
        continue
    try:
        exec(source, env)
    except Exception:
        print('Skipped a cell due to error:')
        traceback.print_exc()

# Ensure functions exist
for name in ['load_data_wrapper', 'init_network', 'learn']:
    if name not in env:
        print(f'Required symbol {name} not found. Aborting.')
        raise SystemExit(1)

load_data_wrapper = env['load_data_wrapper']
init_network = env['init_network']
learn = env['learn']

# Load data
training_data, validation_data, test_data = load_data_wrapper()

# Subsample training and test sets for speed
train_subset = training_data[:1000]
test_subset = list(test_data)[:1000]

nn = init_network([784, 30, 10])

epochs = 3
mini_batch_size = 10
learning_rate = 3.0

print(f'Starting training: epochs={epochs}, train_samples={len(train_subset)}, mini_batch_size={mini_batch_size}\n')
start = time.time()
learn(nn, train_subset, epochs, mini_batch_size, learning_rate, test_data=test_subset)
end = time.time()
print(f"Training finished in {end - start:.2f} seconds ({(end-start)/epochs:.2f} s/epoch)")

# Evaluate on validation set (if present)
try:
    val_acc = 100.0 * env['evaluate'](nn, validation_data) / len(validation_data)
    print(f'Validation accuracy: {val_acc:.2f}%')
except Exception:
    print('Could not evaluate on validation set.')
