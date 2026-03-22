# 🤖 Delivery Robot — Q-Learning Agent

A reinforcement learning agent that learns to navigate a 6×6 grid,
pick up a package, and deliver it to the destination — while avoiding obstacles.

Built with **Q-Learning** and deployed as a **Streamlit web app**.

## 🚀 Live Demo
> _Paste your Streamlit link here after deployment_

## 🧠 How it works
- The robot explores the grid using an **ε-greedy policy**
- Every move updates a **Q-table** using the Bellman equation
- Over time the robot learns the optimal delivery route

## ⚙️ Parameters
| Parameter | Value |
|---|---|
| Grid Size | 6 × 6 |
| Alpha (learning rate) | 0.1 |
| Gamma (discount) | 0.9 |
| Epsilon (exploration) | 0.2 |
| Obstacles | 5 |

## 📦 Run locally
```bash
pip install streamlit numpy
streamlit run app.py
```
