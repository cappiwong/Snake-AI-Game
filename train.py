from agent import RainbowAgent
from game import SnakeGameAI
from helper import plot
import torch

BATCH_SIZE = 128
LR = 0.001

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = SnakeGameAI()
    agent = RainbowAgent(input_size=11, hidden_size=256, output_size=3, n_steps=3)

    while True:
        # Get the current state
        state_old = agent.get_state(game)

        # Get the action
        final_move = agent.get_action(state_old)

        # Perform the action and get the new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Store the transition in the replay buffer
        agent.remember(state_old, final_move, reward, state_new, done)

        # Train the agent on a single transition (short memory)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        if done:
            # Reset the game
            game.reset()
            agent.n_games += 1

            # Train the agent on a batch of transitions (long memory)
            agent.train_long_memory()

            # Update the record if the current score is higher
            if score > record:
                record = score
                # Save the model checkpoint
                torch.save(agent.model.state_dict(), "model.pth")

            # Update the target network every 10 games
            if agent.n_games % 10 == 0:
                agent.update_target_model()

            # Print the results
            print(f"Game {agent.n_games}, Score: {score}, Record: {record}")

            # Track and plot scores
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()

