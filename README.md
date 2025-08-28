# Bomberman Deep RL Starter (DQN & PPO)

This is a **from-scratch Deep RL starter** to build a Bomberman AI with **DQN** and **PPO** in Python + PyTorch.
The environment here is a simplified gridworld with bombs and items — meant as a scaffold. 
You should **map it to your tournament's real server state** (via WebSocket) by adapting `envs/bomber_env.py` and `deploy/ws_inference.py`.

## Folder layout
```
bomberman_deeprl_starter/
├─ envs/
│  ├─ bomber_env.py         # Mini gym-like env
│  └─ wrappers.py           # Optional observation/actions utils
├─ agents/
│  ├─ dqn.py                # DQN (from scratch)
│  └─ ppo.py                # PPO (from scratch)
├─ train/
│  ├─ train_dqn.py          # Train DQN
│  └─ train_ppo.py          # Train PPO
├─ deploy/
│  └─ ws_inference.py       # WebSocket inference bridge (skeleton)
├─ utils/
│  ├─ replay_buffer.py
│  ├─ nets.py
│  └─ common.py
├─ requirements.txt
└─ README.md
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
pip install -r requirements.txt

# Train DQN
python -m train.train_dqn --episodes 5000 --save-path checkpoints/dqn.pt

# Train PPO
python -m train.train_ppo --timesteps 5_000_000 --save-path checkpoints/ppo.pt

# Run inference via WebSocket (adapt the URI and state mapping)
python -m deploy.ws_inference --uri ws://localhost:8080/ws --policy checkpoints/dqn.pt --algo dqn
```

## How to adapt to your tournament
1. **State mapping**: The server probably sends JSON including map grid, bombs (pos, fuse), items, players.  
   Convert that JSON into the env observation (see `envs/bomber_env.py: _encode_obs`).
2. **Action mapping**: Map your model's discrete action to the protocol you must send over WebSocket (see `deploy/ws_inference.py`).
3. **Two-bot coordination**: Instantiate **two policies** and share info via a tiny "blackboard" or give different reward shaping per bot (attacker/collector).

## Tips
- Start with DQN + small conv/MLP; keep the action space compact: [stay, up, down, left, right, place_bomb].
- Reward shaping matters:
  - +1 survive a step, +5 pickup item, +20 damage/kill, -30 die, -0.01 step penalty, -5 self-trapped.
- Curriculum: train on easier maps → gradually add randomness (more crates, more items, smarter enemies).
- For PPO, tune: learning rate (3e-4), clip (0.1–0.2), batch size (8K–64K), GAE-lambda (0.95–0.98).

Good luck & have fun! 🚀

## Tournament integration notes

- Example WebSocket server provided by organizers: `ws://171.251.51.213:5001`.
- The server sends map tiles and positions quantized as integers multiplied by 100 (e.g. tile cell x=5 -> position p.x=500).
- Edit `deploy/ws_inference.py::decode_server_state` to match exact server JSON if it differs. Ensure `--player-id` is passed to mark the 'self' channel when running inference:
  python -m deploy.ws_inference --uri ws://171.251.51.213:5001 --policy checkpoints/dqn.pt --algo dqn --player-id <your-player-id>
