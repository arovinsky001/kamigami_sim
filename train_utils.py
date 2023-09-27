import numpy as np
import torch
from torch.nn import functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import dcn, as_tensor, signed_angle_difference


torch.set_printoptions(precision=4, linewidth=150, sci_mode=False)


def train_from_buffer(agent, replay_buffer, pretrain_samples=500, close_agent=False, set_scale_only=False):
    # n_samples = min(pretrain_samples, replay_buffer.size)
    # states = replay_buffer.states[:n_samples]
    # actions = replay_buffer.actions[:n_samples]
    # next_states = replay_buffer.next_states[:n_samples]

    n_samples = min(pretrain_samples, replay_buffer.get_stored_size())
    all_samples = replay_buffer.get_all_transitions()
    states = all_samples['state'][:n_samples]
    actions = all_samples['action'][:n_samples]
    next_states = all_samples['next_state'][:n_samples]

    # if agent.use_object:
    #     n = 12000
    #     states, actions, next_states = states[:n], actions[:n], next_states[:n]

    if set_scale_only:
        states, actions, next_states = states[:agent.scale_idx], actions[:agent.scale_idx], next_states[:agent.scale_idx]
        for model in agent.models:
            model.set_scalers(states, actions, next_states)
        return

    training_losses, test_losses, train_state, train_action, val_state, val_action, val_next_state = train(
            agent, states, actions, next_states,
            set_scalers=True, close_agent=close_agent,
    )

    if agent.save_agent:
        agent.dump(scale_idx=len(train_state))

    training_losses = np.array(training_losses).squeeze()
    test_losses = np.array(test_losses).squeeze()

    print("\nMIN TEST LOSS EPOCH:", test_losses.argmin())
    print("MIN TEST LOSS:", test_losses.min())

    n_entities = agent.n_robots + agent.use_object
    all_idx = torch.arange(3*n_entities)
    xy_idx = all_idx.reshape((n_entities, 3))[:, :2].reshape(n_entities*2)
    heading_idx = all_idx.reshape((n_entities, 3))[:, 2].reshape(n_entities)

    if len(val_state > 0):
        with torch.no_grad():
            agent.models[-1].eval()
            pred_state_delta = agent.models[-1](val_state, val_action, sample=False, delta=True)

        if agent.scale:
            pred_state_delta = agent.models[-1].unstandardize_output(pred_state_delta)
        pred_next_state = agent.dtu.next_state_from_relative_delta(val_state, pred_state_delta)

        error = torch.empty_like(pred_next_state)
        error[:, xy_idx] = abs(val_next_state - pred_next_state)[:, xy_idx]
        error[:, heading_idx] = abs(signed_angle_difference(val_next_state[:, heading_idx], pred_next_state[:, heading_idx])) * 180 / np.pi

        error_mean, error_std, error_max, error_min = error.mean(axis=0), error.std(axis=0), error.max(axis=0)[0], error.min(axis=0)[0]

        print("\nERROR MEAN:", error_mean)
        print("ERROR STD:", error_std)
        print("ERROR MAX:", error_max)
        print("ERROR MIN:", error_min)

        # idx = [0, 1, 4, 5, 8, 9]
        # th = 0.03
        # norm1 = torch.norm(val_state_delta[:, idx[0:2]], dim=-1)
        # norm2 = torch.norm(val_state_delta[:, idx[2:4]], dim=-1)
        # norm3 = torch.norm(val_state_delta[:, idx[4:6]], dim=-1)
        # ind1 = torch.zeros_like(norm1)
        # ind2 = torch.zeros_like(norm2)
        # ind3 = torch.zeros_like(norm3)
        # ind1[norm1 > th] = 1.
        # ind2[norm2 > th] = 1.
        # ind3[norm3 > th] = 1.

        # ind = (ind1 == 1) | (ind2 == 1) | (ind3 == 1)
        # print("\nBIG ERROR MEAN:", error[ind].mean(axis=0))
        # print("BIG ERROR STD:", error[ind].std(axis=0))
        # print("BIG ERROR MAX:", error[ind].max(axis=0)[0])
        # print("BIG ERROR MIN:", error[ind].min(axis=0)[0])

        # ind = (ind1 == 0) & (ind2 == 0) & (ind3 == 0)
        # print("\nSMALL ERROR MEAN:", error[ind].mean(axis=0))
        # print("SMALL ERROR STD:", error[ind].std(axis=0))
        # print("SMALL ERROR MAX:", error[ind].max(axis=0)[0])
        # print("SMALL ERROR MIN:", error[ind].min(axis=0)[0])

        diffs = torch.empty_like(val_state)
        diffs[:, xy_idx] = abs(val_next_state - val_state)[:, xy_idx]
        diffs[:, heading_idx] = abs(signed_angle_difference(val_next_state[:, heading_idx], val_state[:, heading_idx])) * 180 / np.pi

        diffs_mean, diffs_std, diffs_max, diffs_min = diffs.mean(axis=0), diffs.std(axis=0), diffs.max(axis=0)[0], diffs.min(axis=0)[0]

        print("\nACTUAL MEAN:", diffs_mean)
        print("ACTUAL STD:", diffs_std)
        print("ACTUAL MAX:", diffs_max)
        print("ACTUAL MIN:", diffs_min)

        print(f"\nMEAN RATIO: {error_mean / diffs_mean}")
        print(f"STD RATIO: {error_std / diffs_std}")
        print(f"MAX RATIO: {error_max / diffs_max}")

        with torch.no_grad():
            train_predictions = []
            val_predictions = []

            for model in agent.models:
                model.eval()
                train_pred = model(train_state[:len(val_state)], train_action[:len(val_state)], sample=False, delta=True)
                val_pred = model(val_state, val_action, sample=False, delta=True)
                train_predictions.append(train_pred)
                val_predictions.append(val_pred)

        train_pred_std = torch.stack(train_predictions, dim=0).std(dim=0)
        val_pred_std = torch.stack(val_predictions, dim=0).std(dim=0)
        print("\nMEAN TRAIN PRED STD:", train_pred_std.mean())
        print("MEAN VAL PRED STD:", val_pred_std.mean())

        # import pdb;pdb.set_trace()

    if agent.training_plot:
        fig, axes = plt.subplots(1, 2)

        for ax in axes:
            ax.grid()
            if not agent.dist:
                ax.set_yscale('log')

        if agent.dist:
            plot_training_losses = training_losses
            plot_test_losses = test_losses
        else:
            plot_training_losses = training_losses - training_losses.min() + 1e-3
            plot_test_losses = test_losses - test_losses.min() + 1e-3

        axes[0].plot(np.arange(len(training_losses)), plot_training_losses)
        axes[1].plot(np.arange(-1, len(test_losses)-1), plot_test_losses)

        axes[0].set_title('Training Loss')
        axes[1].set_title('Test Loss')

        axes[0].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')

        fig.set_size_inches(12, 3.5)
        plt.show()

        import pdb;pdb.set_trace()

def train(agent, train_state, train_action, train_next_state, set_scalers=False, close_agent=False):
    train_state, train_action, train_next_state = as_tensor(train_state, train_action, train_next_state)
    og_train_state, og_train_action, og_train_next_state = train_state.clone(), train_action.clone(), train_next_state.clone()

    if agent.validation:
        # import random
        # seed = 2
        # random.seed(seed)
        # torch.random.manual_seed(seed)
        # np.random.seed(seed)

        # n_val = int(len(og_train_state) * 0.2) if close_agent else 0
        n_val = int(len(og_train_state) * 0.2)

        rand_idx = torch.randperm(len(og_train_state))
        train_state, train_action, train_next_state = og_train_state[rand_idx], og_train_action[rand_idx], og_train_next_state[rand_idx]
        val_state, val_action, val_next_state = train_state[:n_val], train_action[:n_val], train_next_state[:n_val]
        train_state, train_action, train_next_state = train_state[n_val:], train_action[n_val:], train_next_state[n_val:]
    else:
        n_val = 0

    for model_idx, model in enumerate(agent.models):
        if not agent.validation:
            rand_idx = torch.randperm(len(og_train_state))
            train_state, train_action, train_next_state = og_train_state[rand_idx], og_train_action[rand_idx], og_train_next_state[rand_idx]
            val_state, val_action, val_next_state = train_state[:n_val], train_action[:n_val], train_next_state[:n_val]
            train_state, train_action, train_next_state = train_state[n_val:], train_action[n_val:], train_next_state[n_val:]

        if set_scalers:
            model.set_scalers(train_state, train_action, train_next_state)

        val_state_delta = agent.dtu.compute_relative_delta_xysc(val_state, val_next_state)
        if agent.scale:
            val_state_delta = model.standardize_output(val_state_delta)
        val_state, val_action, val_next_state = as_tensor(val_state, val_action, val_next_state)

        train_losses, test_losses = [], []
        n_batches = int(np.ceil(len(train_state) / agent.batch_size))
        if n_batches * agent.batch_size > len(train_state) and n_batches != 1:
            if len(train_state) - ((n_batches - 1) * agent.batch_size) < 0.2 * agent.batch_size:
                n_batches -= 1
        # n_batches = max(1, int(np.floor(len(train_state) / batch_size)))

        def evaluate():
            with torch.no_grad():
                model.eval()

                if agent.dist:
                    pred_state_delta_dist = model(val_state, val_action, ret_dist=True)
                    test_loss_mean = -pred_state_delta_dist.log_prob(val_state_delta).mean()
                    # test_loss_mean = F.mse_loss(pred_state_delta_dist.mean, val_state_delta, reduction='mean')
                else:
                    pred_state_delta = model(val_state, val_action, sample=False, delta=True)
                    test_loss_mean = F.mse_loss(pred_state_delta, val_state_delta, reduction='mean')

            test_losses.append(dcn(test_loss_mean))

        evaluate()
        tqdm.write(f"Pre-Train: test loss: {test_losses[-1]}")

        print(f"\n\nTRAINING MODEL {model_idx}\n")
        for i in tqdm(range(-1, agent.train_epochs), desc="Epoch", position=0, leave=False):
            # train
            if agent.meta:
                train_loss_mean = model.update_meta(train_state, train_action, train_next_state)
            else:
                shuffle_idx = np.random.permutation(len(train_state))
                train_state, train_action, train_next_state = train_state[shuffle_idx], train_action[shuffle_idx], train_next_state[shuffle_idx]

                for j in tqdm(range(n_batches), desc="Batch", position=1, leave=False):
                    start, end = j * agent.batch_size, (j + 1) * agent.batch_size
                    batch_state, batch_action, batch_next_state = train_state[start:end], train_action[start:end], train_next_state[start:end]
                    train_loss_mean = model.update(batch_state, batch_action, batch_next_state)

            evaluate()
            train_losses.append(train_loss_mean)

            tqdm.write(f"{i+1}: train loss: {train_losses[-1]:.5f} | test loss: {test_losses[-1]:.5f}")
            tqdm.write(f'model_min_std: {model.min_logvar.exp().sqrt().mean()}')
            tqdm.write(f'model_max_std: {model.max_logvar.exp().sqrt().mean()}')

    if agent.meta:
        for g in model.optimizer.param_groups:
            # g['lr'] = model.update_lr
            g['lr'] = torch.clamp(model.update_lr, 1e-4, 1)
            print("\nUPDATE LEARNING RATE:", model.update_lr)

    model.eval()
    agent.trained = True

    return train_losses, test_losses, train_state, train_action, val_state, val_action, val_next_state
