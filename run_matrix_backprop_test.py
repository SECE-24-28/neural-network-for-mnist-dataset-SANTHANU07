import json
import traceback
import numpy as np

nb_path = 'neural-network.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Execution environment for notebook code
env = {'np': np}

skip_keywords = [
    'load_data_wrapper()',
    'learn(nn,',
    "training_data, validation_data, test_data = load_data_wrapper()",
    'nn = init_network([784, 30, 10])',
    "for file in range(0,10):",
    "recognize_image('",
]

for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    source = ''.join(cell.get('source', []))
    if any(k in source for k in skip_keywords):
        # skip main-running cells that require MNIST or files
        continue
    try:
        exec(source, env)
    except Exception:
        # print the error and continue; some cells may reference missing resources
        print('Skipped a cell due to error:')
        traceback.print_exc()

# Verify functions exist
if 'init_network' not in env or 'stochastic_gradient_descent' not in env:
    print('Required functions not found in executed cells. Aborting.')
    raise SystemExit(1)

# Create a tiny network and a small mini-batch
init_network = env['init_network']
stochastic_gradient_descent = env['stochastic_gradient_descent']

nn = init_network([4, 3, 2])
mini_batch = []
for _ in range(5):
    x = np.random.randn(4, 1)
    y = np.zeros((2, 1))
    y[np.random.randint(0, 2), 0] = 1.0
    mini_batch.append((x, y))

print('Doing one SGD update on tiny network...')
try:
    stochastic_gradient_descent(nn, mini_batch, 0.1)
    print('SGD update successful.')
    for i, (w, b) in enumerate(zip(nn.weights, nn.biases), start=1):
        print(f'Layer {i}: weights shape={w.shape}, biases shape={b.shape}')
except Exception:
    print('SGD update failed:')
    traceback.print_exc()
