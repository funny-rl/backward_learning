import torch as th
import torch.nn as nn

from modules.layer.spectra_attention import CrossAttentionBlock, SetAttentionBlock, QueryKeyBlock


class SPECTra_RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(SPECTra_RNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_allies = args.n_allies
        self.n_enemies = args.n_enemies
        self.n_entities = self.n_agents + self.n_enemies
        self.n_actions = self.args.n_actions
        self.n_head = self.args.n_head
        self.hidden_size = self.args.hidden_size
        self.use_SAQA = self.args.use_SAQA

        self.observer_feats, self.ally_feats, self.enemy_feats = input_shape

        self.observer_feats = self.observer_feats[-1]
        self.ally_feats_dim = self.ally_feats[-1]
        self.enemy_feats_dim = self.enemy_feats[-1]

        self.own_embedding = nn.Linear(self.observer_feats, self.hidden_size)
        self.allies_embedding = nn.Linear(self.ally_feats_dim, self.hidden_size)
        self.enemies_embedding = nn.Linear(self.enemy_feats_dim, self.hidden_size)

        if self.use_SAQA:
            self.entity_attention = CrossAttentionBlock(d=self.hidden_size, h=self.n_head)

        else:
            self.self_attention = SetAttentionBlock(d=self.hidden_size, h=self.n_head)

        self.rnn = nn.GRU(self.hidden_size, self.hidden_size)

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.residual_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.action_layer = nn.Linear(self.hidden_size, self.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.own_embedding.weight.new(1, self.hidden_size).zero_()

    def forward(self, inputs, hidden_state):

        bs, observer_feats, ally_feats, enemy_feats = inputs

        # print(bs, observer_feats.shape, ally_feats.shape, enemy_feats.shape)

        # print(bs, observer_feats.shape, ally_feats.shape, enemy_feats.shape)

        observer_embedding = self.own_embedding(observer_feats)
        ally_embedding = self.allies_embedding(ally_feats)
        enemy_embedding = self.enemies_embedding(enemy_feats)
        embeddings = th.cat(
            (observer_embedding, ally_embedding, enemy_embedding), dim=1
        )  # (bs * n_agents, n_entities, hidden_size)

        if self.use_SAQA:
            attentive_embedding = self.entity_attention(
                embeddings[:, 0, :].unsqueeze(1), embeddings
            )  # (bs * n_agents, 1, hidden_size)
        else:
            attentive_embedding = self.self_attention(embeddings).mean(dim=1, keepdim=True).squeeze(1)

        attentive_embedding = attentive_embedding.squeeze(1)
        hidden_state = hidden_state.reshape(-1, self.hidden_size)

        attentive_embedding = attentive_embedding.reshape(bs, self.n_agents, self.hidden_size)

        output, hidden_state = self.rnn(attentive_embedding)

        rec_att_embed = output.reshape(-1, self.hidden_size)
        # hidden_state = hidden_state.reshape(bs, self.n_agents, self.hidden_size)

        q = self.layer_norm(rec_att_embed + self.residual_layer(rec_att_embed))
        q = self.action_layer(q)
        return q, hidden_state
