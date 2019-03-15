# assignment for the course
# http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html

import random
from collections import defaultdict
from itertools import izip


class Easy21Env:
  TERMINAL = None, None
  ACTIONS = ['stick', 'hit']
  PROB_NEG = 1./3. #0.0

  def __init__(self, dealer_stick=17):
    self.dealer_stick = dealer_stick

  def reset(self):
    return self.draw_card(black=True), self.draw_card(black=True)

  def step(self, state, action):
    assert action in Easy21Env.ACTIONS
    dealer_card, player_sum = state
    if action == 'hit':
      player_sum += self.draw_card()
      if self.check_bust(player_sum):
        return Easy21Env.TERMINAL, -1.0
      else:
        return (dealer_card, player_sum), 0.0
    else:
      dealer_sum = dealer_card
      while True:
        if dealer_sum >= self.dealer_stick:
          if dealer_sum > player_sum:
            reward = -1.0
          elif dealer_sum == player_sum:
            reward = 0.0
          else:
            reward = 1.0
          return Easy21Env.TERMINAL, reward
        else:
          dealer_sum += self.draw_card()
          if self.check_bust(dealer_sum):
            return Easy21Env.TERMINAL, 1.0

  def draw_card(self, black=False):
    value = random.randint(1, 10)
    sign = -1 if random.random() < Easy21Env.PROB_NEG else 1
    ans = sign * value if not black else value
    return ans

  def check_bust(self, card_sum):
    return card_sum > 21 or card_sum < 1


class SimplePolicy:
  def __init__(self, player_stick):
    self.player_stick = player_stick

  def act(self, state):
    dealer_card, player_sum = state
    return 'stick' if player_sum >= self.player_stick else 'hit'


class EpsilonGreedyPolicy:
  DEFAULT_COUNT = 100.0
  DEFAULT_EPSILON = 0.05

  def __init__(self, actions, q, counts):
    self.actions = actions
    self.q = q
    self.counts = counts

  def act(self, state, greedy=False, epsilon=None):
    if greedy:
      return self.act_greedy(state)
    else:
      if self.counts is not None:
        state_count = sum(self.counts[(state, action)]
                          for action in self.actions)
        epsilon = (EpsilonGreedyPolicy.DEFAULT_COUNT /
                   (EpsilonGreedyPolicy.DEFAULT_COUNT + state_count))
      else:
        epsilon = EpsilonGreedyPolicy.DEFAULT_EPSILON
      if random.random() < epsilon:
        return random.choice(self.actions)
      return self.act_greedy(state)

  def act_greedy(self, state):
    best_q = -float('inf')
    best_action = None
    for action in self.actions:
      curr_q = self.q[(state, action)]
      if curr_q > best_q:
        best_q = curr_q
        best_action = action
    return best_action


def test_env():
  NUMBER_OF_TRIALS = 10000
  DEALER_STICK = 17
  env = Easy21Env(DEALER_STICK)
  policy = SimplePolicy(DEALER_STICK)
  reward_sum = 0
  for _ in xrange(NUMBER_OF_TRIALS):
    state = env.reset()
    while state != Easy21Env.TERMINAL:
      state, reward = env.step(state, policy.act(state))
    reward_sum += reward
  average_reward = reward_sum / float(NUMBER_OF_TRIALS)
  print 'average_reward:', average_reward


def plot_q(q, actions):
  START_DEALER_CARD = 1
  END_DEALER_CARD = 11
  START_PLAYER_SUM = 1 #12
  END_PLAYER_SUM = 22
  print 'V:'
  for dealer_card in xrange(START_DEALER_CARD, END_DEALER_CARD):
    row = []
    for player_sum in xrange(START_PLAYER_SUM, END_PLAYER_SUM):
      state = (dealer_card, player_sum)
      row += [max(q[(state, action)] for action in actions)]
    print ' '.join(['{:+03.2f}'.format(val) for val in row])
  print 'Q(, hit):'
  for dealer_card in xrange(START_DEALER_CARD, END_DEALER_CARD):
    row = []
    for player_sum in xrange(START_PLAYER_SUM, END_PLAYER_SUM):
      state = (dealer_card, player_sum)
      row += [q[(state, 'hit')]]
    print ' '.join(['{:+03.2f}'.format(val) for val in row])
  print 'Q(, stick):'
  for dealer_card in xrange(START_DEALER_CARD, END_DEALER_CARD):
    row = []
    for player_sum in xrange(START_PLAYER_SUM, END_PLAYER_SUM):
      state = (dealer_card, player_sum)
      row += [q[(state, 'stick')]]
    print ' '.join(['{:+03.2f}'.format(val) for val in row])
  print 'Decision:'
  for dealer_card in xrange(START_DEALER_CARD, END_DEALER_CARD):
    row = []
    for player_sum in xrange(START_PLAYER_SUM, END_PLAYER_SUM):
      state = (dealer_card, player_sum)
      row += ['s' if q[(state, 'stick')] > q[(state, 'hit')] else 'h']
    print ' '.join(row)


def evaluate(env, policy):
  NUMBER_OF_EVAL_TRIALS = 10000
  env = Easy21Env()
  reward_sum = 0
  for _ in xrange(NUMBER_OF_EVAL_TRIALS):
    state = env.reset()
    while state != Easy21Env.TERMINAL:
      state, reward = env.step(state, policy.act(state, greedy=True))
    reward_sum += reward
  average_reward = reward_sum / float(NUMBER_OF_EVAL_TRIALS)
  print 'average_reward:', average_reward


def monte_carlo(episodes):
  env = Easy21Env()
  q = defaultdict(int)
  counts = defaultdict(int)
  policy = EpsilonGreedyPolicy(Easy21Env.ACTIONS, q, counts)
  for _ in xrange(episodes):
    state = env.reset()
    states = []
    actions = []
    rewards = []
    while state != Easy21Env.TERMINAL:
      states += [state]
      actions += [policy.act(state)]
      state, reward = env.step(state, actions[-1])
      rewards += [reward]
    update_monte_carlo(q, counts, states, actions, rewards)
  plot_q(q, Easy21Env.ACTIONS)
  # plot_q(counts, Easy21Env.ACTIONS)
  evaluate(env, policy)


def update_monte_carlo(q, counts, states, actions, rewards):
  reward = rewards[-1]
  for state, action in izip(states, actions):
    counts[(state, action)] += 1
    step = 1.0 / counts[(state, action)]
    q[(state, action)] = (q[(state, action)] +
                          step * (reward - q[(state, action)]))


def sarsa(lam, episodes):
  env = Easy21Env()
  q = defaultdict(int)
  counts = defaultdict(int)
  policy = EpsilonGreedyPolicy(Easy21Env.ACTIONS, q, counts)
  for _ in xrange(episodes):
    s = env.reset()
    a = policy.act(s)
    e = defaultdict(int)
    while s != Easy21Env.TERMINAL:
      new_s, r = env.step(s, a)
      new_a = policy.act(new_s)
      delta = r + q[(new_s, new_a)] - q[(s, a)]
      e[(s, a)] += 1
      counts[(s, a)] += 1
      update_sarsa(q, counts, e, delta, lam)
      s, a = new_s, new_a
  plot_q(q, Easy21Env.ACTIONS)
  evaluate(env, policy)


def update_sarsa(q, counts, e, delta, lam):
  for s, a in e:
    step = 1.0 / counts[(s, a)]
    q[(s, a)] += step * delta * e[(s, a)]
    e[(s, a)] *= lam


class Q:
  def __init__(self, actions):
    self.param = [0] * self.len_features()
    self.actions = actions

  def __getitem__(self, key):
    s, a = key
    features = self.encode(s, a)
    return sum(first * second for first, second in izip(features,
                                                        self.param))

  def grad(self, s, a):
    return self.encode(s, a)

  def update(self, delta_param):
    self.param = [first + second for first, second in izip(self.param,
                                                           delta_param)]

  def encode(self, s, a):
    dc, ps = s
    features = []
    for dc_lower, dc_higher in [(1, 4), (5, 7), (8, 10)]:
      for ps_lower, ps_higher in [(1, 4), (5, 8), (9, 12),
                                  (13, 15), (16, 18), (19, 21)]:
        for curr_a in self.actions:
          if (dc >= dc_lower and dc <= dc_higher and
              ps >= ps_lower and ps <= ps_higher and
              a == curr_a):
            features += [1]
          else:
            features += [0]
    return features

  def len_features(self):
    return 36

  def get_init_eligibility(self):
    return [0] * self.len_features()


def sarsa_approximation(lam, step, episodes):
  env = Easy21Env()
  q = Q(Easy21Env.ACTIONS)
  policy = EpsilonGreedyPolicy(Easy21Env.ACTIONS, q, None)
  for _ in xrange(episodes):
    s = env.reset()
    a = policy.act(s)
    e = q.get_init_eligibility()
    while s != Easy21Env.TERMINAL:
      new_s, r = env.step(s, a)
      new_a = policy.act(new_s)
      delta = r + q[(new_s, new_a)] - q[(s, a)]
      grad = q.grad(s, a)
      e = [lam * old + deriv for old, deriv in izip(e, grad)]
      q.update([step * delta * val for val in e])
      s, a = new_s, new_a
  plot_q(q, Easy21Env.ACTIONS)
  evaluate(env, policy)


if __name__ == '__main__':
  # monte_carlo(10000) #10000000)
  # sarsa(0.9, 10000)
  sarsa_approximation(0.9, 0.01, 10000)

