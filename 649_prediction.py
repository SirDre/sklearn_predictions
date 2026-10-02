import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from datetime import datetime
from math import comb
from itertools import combinations
import random

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
PREDICTED_DATE = "03-Oct-26"
FILE_PATH = "results.csv"              # Date, Numbers  (Numbers = combination id)
REFERENCE_PATH = "649Results2020.txt"  # FREQ, RESULTS, DATE  (id -> "n1 n2 n3 n4 n5 n6")

# 6/49 lottery definition
BALLS_IN_POOL = 49          # n  - number of balls in the pool
BALLS_DRAWN = 6             # r  - number of balls to be drawn
NUMBER_OF_MATCHES = 6       # m  - expected number of matches (jackpot)

# Bonus ball type 1: taken from the remaining pool (7th ball drawn from the 43 left)
BONUS_REMAINING_MATCHES = BALLS_DRAWN - 1   # default m-1  -> "5 + bonus"

# Bonus ball type 2: taken from a separate bonus pool
BONUS_POOL_BALLS_DRAWN = 1      # balls to be drawn from the bonus pool
BONUS_POOL_MATCHES = 1          # matches with bonus pool
BONUS_POOL_SIZE = BALLS_IN_POOL # balls in the bonus pool (may equal the main pool)

TOTAL_COMBINATIONS = comb(BALLS_IN_POOL, BALLS_DRAWN)   # C(49, 6) = 13,983,816

# Number of combinations printed by each selection method
SELECTION_COUNT = 5
QUICK_PICK_SEED = None   # set to an int for repeatable quick picks


# ---------------------------------------------------------------------------
# Combination <-> id mapping (lexicographic rank, 1-based) - the "FREQ" id
# ---------------------------------------------------------------------------
def combination_to_id(combination, n=BALLS_IN_POOL, r=BALLS_DRAWN) -> int:
    """
    Map a sorted r-combination of 1..n to its 1-based lexicographic rank.
    e.g. (1, 2, 3, 4, 13, 48) -> 359   (matches REFERENCE_PATH)

    Parameters:
        combination (iterable[int]): r distinct numbers in 1..n
        n (int): balls in the pool
        r (int): balls drawn

    Returns:
        int: combination id in 1..C(n, r)
    """
    combination = sorted(int(v) for v in combination)
    if len(combination) != r or combination[0] < 1 or combination[-1] > n:
        raise ValueError(f"combination must be {r} distinct numbers in 1..{n}: {combination}")
    rank = 1
    previous = 0
    for position, value in enumerate(combination):
        remaining = r - position - 1
        for skipped in range(previous + 1, value):
            rank += comb(n - skipped, remaining)
        previous = value
    return rank


def id_to_combination(combination_id: int, n=BALLS_IN_POOL, r=BALLS_DRAWN) -> tuple:
    """
    Inverse of combination_to_id: map a 1-based rank to its sorted r-combination.
    e.g. 359 -> (1, 2, 3, 4, 13, 48)

    Parameters:
        combination_id (int): id in 1..C(n, r)
        n (int): balls in the pool
        r (int): balls drawn

    Returns:
        tuple[int]: the r numbers of the combination
    """
    total = comb(n, r)
    if not 1 <= combination_id <= total:
        raise ValueError(f"id must be in 1..{total:,}: {combination_id}")
    remaining_rank = combination_id - 1
    combination = []
    value = 1
    for position in range(r):
        remaining = r - position - 1
        while True:
            block = comb(n - value, remaining)   # combos starting with `value` here
            if remaining_rank < block:
                break
            remaining_rank -= block
            value += 1
        combination.append(value)
        value += 1
    return tuple(combination)


def format_combination(combination) -> str:
    """Format a combination as space separated numbers, like the RESULTS column."""
    return " ".join(str(v) for v in combination)


# ---------------------------------------------------------------------------
# Lottery odds formula:  x = C(n, r) / ( C(r, m) * C(n - r, r - m) )
# ---------------------------------------------------------------------------
def lottery_odds(n=BALLS_IN_POOL, r=BALLS_DRAWN, m=NUMBER_OF_MATCHES) -> float:
    """
    Odds (1 to x) of matching exactly m of the r balls drawn from a pool of n.

    Parameters:
        n (int): number of balls in the pool
        r (int): number of balls to be drawn
        m (int): expected number of matches

    Returns:
        float: x, meaning the odds are 1 to x
    """
    return comb(n, r) / (comb(r, m) * comb(n - r, r - m))


def bonus_remaining_pool_odds(n=BALLS_IN_POOL, r=BALLS_DRAWN, m=BONUS_REMAINING_MATCHES) -> float:
    """
    Bonus ball taken from the remaining pool: one extra ball is drawn from the
    n - r balls left over.  Odds of matching m main numbers AND the bonus ball
    (default m = r - 1, i.e. "5 + bonus" for a 6/49 game).

    Favourable tickets: choose m of the r winning numbers, the bonus ball itself
    (1 way), and the remaining r - m - 1 numbers from the n - r - 1 losing balls.
    """
    favourable = comb(r, m) * 1 * comb(n - r - 1, r - m - 1)
    return comb(n, r) / favourable


def bonus_separate_pool_odds(n=BALLS_IN_POOL, r=BALLS_DRAWN, m=NUMBER_OF_MATCHES,
                             bonus_drawn=BONUS_POOL_BALLS_DRAWN,
                             bonus_matches=BONUS_POOL_MATCHES,
                             bonus_pool=BONUS_POOL_SIZE) -> float:
    """
    Bonus ball taken from a separate pool (a number may come up twice).
    Odds of m main matches AND bonus_matches of the bonus_drawn bonus balls.
    The two draws are independent, so the odds multiply.
    """
    main = lottery_odds(n, r, m)
    bonus = lottery_odds(bonus_pool, bonus_drawn, bonus_matches)
    return main * bonus


def print_odds_table() -> None:
    """Print the full 6/49 odds table, including both bonus ball types."""
    print(f"\n6/49 lottery: {BALLS_DRAWN} balls drawn from {BALLS_IN_POOL}, "
          f"C({BALLS_IN_POOL},{BALLS_DRAWN}) = {TOTAL_COMBINATIONS:,} combinations")
    print("Formula: x = C(n,r) / ( C(r,m) * C(n-r, r-m) )   ->  odds are 1 to x\n")
    rows = [(f"{m} of {BALLS_DRAWN}", lottery_odds(m=m)) for m in range(BALLS_DRAWN, 1, -1)]
    rows.append((f"{BONUS_REMAINING_MATCHES} + bonus (remaining pool)", bonus_remaining_pool_odds()))
    rows.append((f"{NUMBER_OF_MATCHES} + bonus (separate pool of {BONUS_POOL_SIZE})", bonus_separate_pool_odds()))
    print(f"{'Matches (m)':<36}{'Odds 1 to x':>18}")
    for label, odds in rows:
        print(f"{label:<36}{odds:>18,.2f}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_and_prepare_data(file_path) -> DataFrame:
    """
    Load results.csv and map every combination id to its 6 numbers.

    Parameter:
        file_path (str): Path to the CSV file (Date, Numbers)

    Returns:
        DataFrame: date, id, and one column per ball (b1..b6)
    """
    df = pd.read_csv(file_path, dtype={'Date': 'object'})
    df['date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    df = df.sort_values('date').reset_index(drop=True)
    df['Numbers'] = df['Numbers'].astype(int)

    combos = [id_to_combination(i) for i in df['Numbers']]
    for k in range(BALLS_DRAWN):
        df[f'b{k + 1}'] = [c[k] for c in combos]
    df['Combination'] = [format_combination(c) for c in combos]
    return df


def load_reference(reference_path) -> dict:
    """
    Load 649Results2020.txt (FREQ, RESULTS, DATE) as an id -> combination lookup
    and verify every row against the ranking formula.

    Returns:
        dict: {id: "n1 n2 n3 n4 n5 n6"}
    """
    try:
        ref = pd.read_csv(reference_path)
    except FileNotFoundError:
        print(f"Reference file {reference_path} not found - skipping verification.")
        return {}
    ref.columns = [c.strip() for c in ref.columns]
    ref['RESULTS'] = ref['RESULTS'].str.strip()
    mismatches = sum(
        combination_to_id(res.split()) != int(freq)
        for freq, res in zip(ref['FREQ'], ref['RESULTS'])
    )
    print(f"Reference {reference_path}: {len(ref):,} rows, "
          f"{mismatches} id/combination mismatches against the ranking formula.")
    return dict(zip(ref['FREQ'].astype(int), ref['RESULTS']))


# ---------------------------------------------------------------------------
# "Best possibility" analysis
# ---------------------------------------------------------------------------
def ball_frequencies(df: DataFrame) -> np.ndarray:
    """
    Count how often each ball 1..49 appeared in the historical draws.

    Returns:
        numpy.array: index 0 is ball 1, ..., index 48 is ball 49
    """
    balls = df[[f'b{k + 1}' for k in range(BALLS_DRAWN)]].values.ravel()
    return np.bincount(balls, minlength=BALLS_IN_POOL + 1)[1:]


def match_count(combination_a, combination_b) -> int:
    """Number of balls shared by two combinations."""
    return len(set(combination_a) & set(combination_b))


def best_possibility(frequencies: np.ndarray, r=BALLS_DRAWN,
                     selection_count=SELECTION_COUNT, candidate_pool_size=20,
                     max_shared_balls=4, quick_pick_range=None,
                     excluded_ids=(), seed=QUICK_PICK_SEED) -> list:
    """
    Rank diverse combinations from the historically most frequent balls.

    Every r-combination of the `candidate_pool_size` most frequent balls is
    scored by its total frequency.  Selections are taken best-score first, and a
    candidate is skipped when it shares more than `max_shared_balls` balls with
    one already selected, so the `selection_count` (default 5) results are
    distinct combinations rather than one-ball variations of the same ticket.

    When `quick_pick_range` is given as (low_id, high_id), `selection_count`
    quick picks from that range are appended after the frequency-ranked ones,
    skipping `excluded_ids` and the ids already selected.

    Returns:
        list: (combination tuple, combination id, total frequency score) entries;
              frequency-ranked first, then any quick picks
    """
    if len(frequencies) != BALLS_IN_POOL:
        raise ValueError(f"frequencies must contain {BALLS_IN_POOL} values")
    if not 1 <= r <= candidate_pool_size <= BALLS_IN_POOL:
        raise ValueError("expected 1 <= r <= candidate_pool_size <= BALLS_IN_POOL")
    if selection_count < 1 or not 0 <= max_shared_balls < r:
        raise ValueError("selection_count must be positive and max_shared_balls less than r")

    order = np.argsort(-frequencies, kind='stable')        # highest count first
    candidate_balls = sorted(int(i) + 1 for i in order[:candidate_pool_size])
    candidates = []
    for combination in combinations(candidate_balls, r):
        score = sum(frequencies[ball - 1] for ball in combination)
        candidates.append((combination, combination_to_id(combination), int(score)))
    candidates.sort(key=lambda candidate: (-candidate[2], candidate[0]))

    selections = []
    seen_ids = set()
    for candidate in candidates:
        if candidate[1] in seen_ids:
            continue
        if all(match_count(candidate[0], selected[0]) <= max_shared_balls
               for selected in selections):
            selections.append(candidate)
            seen_ids.add(candidate[1])
            if len(selections) == selection_count:
                break

    if quick_pick_range is not None:
        excluded = set(excluded_ids) | seen_ids
        selections += quick_pick(*quick_pick_range, excluded, selection_count,
                                 frequencies, seed)

    return selections


# ---------------------------------------------------------------------------
# "Mirror the latest draw" analysis (left / right step approach)
# ---------------------------------------------------------------------------
def _neighbours(ball: int, n=BALLS_IN_POOL) -> list:
    """Balls directly left and right of `ball` that exist in the pool."""
    return [b for b in (ball - 1, ball + 1) if 1 <= b <= n]


def level_gap(frequencies: np.ndarray, ball: int) -> int:
    """
    How close a ball's frequency is to its nearest-frequency neighbour.
    0 means it sits level with the ball next to it (like 4 next to 5).
    """
    own = int(frequencies[ball - 1])
    return min(abs(own - int(frequencies[b - 1])) for b in _neighbours(ball))


def dip_depth(frequencies: np.ndarray, ball: int) -> float:
    """
    How far a ball's frequency sits below the average of its neighbours.
    Positive = a "dip" between higher frequencies, i.e. a ball catching up
    (like 12 between 11 and 13).
    """
    around = [int(frequencies[b - 1]) for b in _neighbours(ball)]
    return sum(around) / len(around) - int(frequencies[ball - 1])


def mirror_combination(frequencies: np.ndarray, last_draw, anchor: int,
                       window=2, avoid_last_draw=True,
                       n=BALLS_IN_POOL, r=BALLS_DRAWN) -> tuple:
    """
    Build one combination that mirrors the shape of the latest draw around a
    fixed 2nd number (`anchor`), using the left / right step approach:

      1st  - step LEFT of the anchor by the latest draw's first gap, then take
             the ball in the +/- `window` whose frequency is most level with
             its neighbour.
      2nd  - the fixed anchor.
      3rd  - step RIGHT by the latest draw's second gap, then take the deepest
             "dip" in the +/- `window` (a ball catching up with both neighbours).
      4th  - jump RIGHT: the ball whose frequency is closest to the 3rd pick's,
             ties broken by distance to the latest draw's third gap.
      rest - forced right: the deepest dips between higher frequencies.

    Balls of the latest draw are skipped (except the anchor) when
    `avoid_last_draw` is True, as long as enough other balls remain.

    Returns:
        tuple[int]: the sorted r-combination with `anchor` as its 2nd number
    """
    if r < 4:
        raise ValueError("the mirror approach needs at least 4 balls per draw")
    tail = r - 4                                   # picks after the 4th number
    if not 2 <= anchor <= n - (r - 2):
        raise ValueError(f"anchor {anchor} cannot be the 2nd of {r} numbers in 1..{n}")

    last_draw = sorted(int(v) for v in last_draw)
    gaps = [b - a for a, b in zip(last_draw, last_draw[1:])]
    blocked = (set(last_draw) - {anchor}) if avoid_last_draw else set()

    def freq(ball):
        return int(frequencies[ball - 1])

    def usable(low, high, needed=1):
        """Balls in low..high, skipping the latest draw's when enough remain."""
        everything = list(range(low, high + 1))
        allowed = [b for b in everything if b not in blocked]
        return allowed if len(allowed) >= needed else everything

    # 1st: step left, most level with its neighbour
    target = max(1, anchor - gaps[0])
    candidates = usable(max(1, target - window), min(anchor - 1, target + window))
    first = min(candidates,
                key=lambda b: (level_gap(frequencies, b), abs(b - target), b))

    # 3rd: step right, deepest dip (catching up with both neighbours)
    limit = n - tail - 1
    target = min(anchor + gaps[1], limit)
    candidates = usable(max(anchor + 1, target - window), min(limit, target + window))
    third = min(candidates,
                key=lambda b: (-dip_depth(frequencies, b), abs(b - target), b))

    # 4th: jump right, frequency closest to the 3rd pick
    limit = n - tail
    target = min(third + gaps[2], limit)
    candidates = usable(third + 1, limit)
    fourth = min(candidates,
                 key=lambda b: (abs(freq(b) - freq(third)), abs(b - target), b))

    # rest: forced right, deepest dips between higher frequencies
    candidates = usable(fourth + 1, n, needed=tail)
    deepest = sorted(candidates, key=lambda b: (-dip_depth(frequencies, b), b))[:tail]

    return tuple([first, anchor, third, fourth] + sorted(deepest))


def mirror_latest_draw(frequencies: np.ndarray, last_draw,
                       selection_count=SELECTION_COUNT, window=2,
                       avoid_last_draw=True,
                       n=BALLS_IN_POOL, r=BALLS_DRAWN) -> list:
    """
    Mirror the latest draw once for each of the `selection_count` (default 5)
    most frequently drawn balls, using that ball as the fixed 2nd number.
    A different 2nd number guarantees the combinations are distinct.

    A frequent ball that cannot be a 2nd number (1, or too close to the top of
    the pool) is skipped in favour of the next most frequent one.

    Returns:
        list: (combination tuple, combination id, total frequency score) entries,
              in anchor order (most frequent anchor first)
    """
    if len(frequencies) != n:
        raise ValueError(f"frequencies must contain {n} values")
    if selection_count < 1:
        raise ValueError("selection_count must be positive")

    order = np.argsort(-frequencies, kind='stable')        # highest count first
    anchors = [int(i) + 1 for i in order if 2 <= int(i) + 1 <= n - (r - 2)]

    selections = []
    seen_ids = set()
    for anchor in anchors:
        combination = mirror_combination(frequencies, last_draw, anchor,
                                         window, avoid_last_draw, n, r)
        combination_id = combination_to_id(combination, n, r)
        if combination_id in seen_ids:
            continue
        seen_ids.add(combination_id)
        score = int(sum(frequencies[ball - 1] for ball in combination))
        selections.append((combination, combination_id, score))
        if len(selections) == selection_count:
            break
    return selections


# ---------------------------------------------------------------------------
# Quick pick (random combination id within the mirror-pick id range)
# ---------------------------------------------------------------------------
def quick_pick(low_id: int, high_id: int, excluded_ids=(),
               selection_count=SELECTION_COUNT, frequencies: np.ndarray = None,
               seed=QUICK_PICK_SEED, n=BALLS_IN_POOL, r=BALLS_DRAWN) -> list:
    """
    Pick `selection_count` (default 5) distinct random combination ids between
    `low_id` and `high_id` inclusive, skipping every id in `excluded_ids`
    (e.g. ids already drawn in FILE_PATH and REFERENCE_PATH).

    Every id in the range is equally likely, like a lottery terminal quick pick.

    Returns:
        list: (combination tuple, combination id, total frequency score) entries,
              sorted by combination id. Score is 0 when no frequencies are given.
    """
    total = comb(n, r)
    low_id, high_id = sorted((int(low_id), int(high_id)))
    if not 1 <= low_id <= high_id <= total:
        raise ValueError(f"id range must lie within 1..{total:,}: {low_id:,}..{high_id:,}")
    if selection_count < 1:
        raise ValueError("selection_count must be positive")

    excluded = {int(i) for i in excluded_ids if low_id <= int(i) <= high_id}
    available = (high_id - low_id + 1) - len(excluded)
    if available < selection_count:
        raise ValueError(f"only {available} unused ids between {low_id:,} and {high_id:,}")

    rng = random.Random(seed)
    picked = set()
    while len(picked) < selection_count:
        candidate = rng.randint(low_id, high_id)
        if candidate not in excluded:
            picked.add(candidate)

    selections = []
    for combination_id in sorted(picked):
        combination = id_to_combination(combination_id, n, r)
        score = (int(sum(frequencies[ball - 1] for ball in combination))
                 if frequencies is not None else 0)
        selections.append((combination, combination_id, score))
    return selections


def quick_pick_from_mirror(mirror_selections: list, excluded_ids=(),
                           selection_count=SELECTION_COUNT,
                           frequencies: np.ndarray = None,
                           seed=QUICK_PICK_SEED) -> list:
    """
    Quick pick between the lowest and highest ids of the mirror-of-latest-draw
    selections, avoiding ids already drawn and the mirror ids themselves.
    """
    if not mirror_selections:
        raise ValueError("mirror_selections is empty")
    mirror_ids = [selection[1] for selection in mirror_selections]
    excluded = set(excluded_ids) | set(mirror_ids)
    return quick_pick(min(mirror_ids), max(mirror_ids), excluded,
                      selection_count, frequencies, seed)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_labels(ax, x, y, color):
    """
    Add value labels to plot points

    Parameters:
        ax: matplotlib axes
        x: x-coordinates (dates)
        y: y-coordinates (values)
        color: label color (string)
    """
    for xi, yi in zip(x, y):
        ax.annotate(f'{int(yi):,}', (xi, yi), textcoords="offset points",
                    xytext=(0, 10), ha='center', va='bottom', color=color, fontsize=7)


def plot_results(df: DataFrame, frequencies: np.ndarray, next_date: datetime,
                 predicted_combination: tuple, predicted_id: int,
                 mirror_selections: list = (), quick_selections: list = ()) -> None:
    """
    Two-panel visualization:
      top    - combination id of every draw over time + predicted id at next_date
               (green star = best possibility, orange diamonds = mirror picks)
      bottom - how often each ball 1..49 was drawn, predicted balls highlighted

    Parameters:
        df (DataFrame): prepared results
        frequencies (numpy.array): draw count per ball
        next_date (datetime): future prediction date
        predicted_combination (tuple): best-possibility combination
        predicted_id (int): its combination id
        mirror_selections (list): entries from mirror_latest_draw (optional)
        quick_selections (list): quick-pick entries (optional)
    """
    fig, (ax_ids, ax_freq) = plt.subplots(2, 1, figsize=(15, 10),
                                          gridspec_kw={'height_ratios': [3, 2]})

    # --- Top: combination id per draw --------------------------------------
    ax_ids.plot(df['date'], df['Numbers'], color='purple', marker='o',
                markersize=3, linewidth=0.8, label='Actual combination id')
    ax_ids.axhline(TOTAL_COMBINATIONS, color='grey', linestyle=':',
                   label=f'C(49,6) = {TOTAL_COMBINATIONS:,}')
    ax_ids.axvline(x=next_date, color='red', linestyle='--',
                   label=f'Predicted date {next_date.strftime("%d-%b-%y")}')
    ax_ids.plot([next_date], [predicted_id], color='green', marker='*', markersize=14,
                linestyle='none',
                label=f'Best possibility id {predicted_id:,} -> {format_combination(predicted_combination)}')
    ax_ids.annotate(f'{predicted_id:,}', (next_date, predicted_id),
                    textcoords="offset points", xytext=(0, 12), ha='center', color='green')
    if mirror_selections:
        mirror_ids = [selection[1] for selection in mirror_selections]
        ax_ids.plot([next_date] * len(mirror_ids), mirror_ids, color='darkorange',
                    marker='D', markersize=6, linestyle='none',
                    label=f'Mirror of latest draw ({len(mirror_ids)} picks)')
    if quick_selections:
        quick_ids = [selection[1] for selection in quick_selections]
        ax_ids.plot([next_date] * len(quick_ids), quick_ids, color='teal',
                    marker='x', markersize=8, linestyle='none',
                    label=f'Quick picks ({len(quick_ids)})')
    # label the last few actual draws only, to keep it readable
    plot_labels(ax_ids, df['date'].tail(5), df['Numbers'].tail(5), 'purple')

    ax_ids.set_xlabel('Date')
    ax_ids.set_ylabel('Combination id (1 .. C(49,6))')
    ax_ids.set_title('6/49 draw results as combination ids, with best-possibility prediction')
    ax_ids.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=8)
    ax_ids.grid(True)
    ax_ids.tick_params(axis='x', rotation=45)

    # --- Bottom: ball frequency --------------------------------------------
    balls = np.arange(1, BALLS_IN_POOL + 1)
    colors = ['green' if b in predicted_combination else 'steelblue' for b in balls]
    ax_freq.bar(balls, frequencies, color=colors)
    ax_freq.axhline(frequencies.mean(), color='red', linestyle='--',
                    label=f'mean {frequencies.mean():.1f}')
    for b, f in zip(balls, frequencies):
        ax_freq.annotate(str(f), (b, f), textcoords="offset points", xytext=(0, 2),
                         ha='center', fontsize=7)
    ax_freq.set_xticks(balls)
    ax_freq.set_xlabel('Ball')
    ax_freq.set_ylabel('Times drawn')
    ax_freq.set_title(f'Ball frequency over {len(df)} draws '
                      f'(green = {BALLS_DRAWN} most frequent = best possibility)')
    ax_freq.legend(loc='upper right', fontsize=8)
    ax_freq.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('649_predictions.png', dpi=120)
    plt.show()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_selection_table(selections: list, show_anchor=False) -> None:
    """Print a ranked table of (combination, id, score) selections."""
    anchor_header = f"{'Anchor (2nd)':<14}" if show_anchor else ""
    print(f"  {'Rank':<6}{anchor_header}{'Numbers':<22}{'Combination id':>16}{'Frequency score':>19}")
    for rank, (combination, combination_id, score) in enumerate(selections, start=1):
        anchor = f"{combination[1]:<14}" if show_anchor else ""
        print(f"  {rank:<6}{anchor}{format_combination(combination):<22}"
              f"{combination_id:>16,}{score:>19,}")


def print_predictions(df: DataFrame, reference: dict, frequencies: np.ndarray,
                      next_date: datetime, selections: list,
                      mirror_selections: list, quick_selections: list = (),
                      excluded_count: int = 0) -> None:
    """
    Print the id -> combination mapping of recent draws, the frequency-ranked
    selections, the mirror-of-latest-draw selections, and the odds.
    """
    print("\nLast 10 draws (id -> combination):")
    for _, row in df.tail(10).iterrows():
        in_ref = f"  [in {REFERENCE_PATH}: {reference[row['Numbers']]}]" if row['Numbers'] in reference else ""
        print(f"  {row['Date']:<10} {row['Numbers']:>12,}  ->  {row['Combination']}{in_ref}")

    seen_in_reference = df['Numbers'].isin(reference.keys()).sum() if reference else 0
    print(f"\n{seen_in_reference} of {len(df)} result ids also appear in {REFERENCE_PATH}.")

    order = np.argsort(-frequencies, kind='stable')
    print("\nMost frequently drawn balls:",
          ", ".join(f"{int(i) + 1} ({frequencies[i]}x)" for i in order[:BALLS_DRAWN]))
    print("Least frequently drawn balls:",
          ", ".join(f"{int(i) + 1} ({frequencies[i]}x)" for i in order[-BALLS_DRAWN:]))

    date_label = next_date.strftime('%d-%b-%y')
    last = df.iloc[-1]
    last_combo = tuple(int(last[f'b{k + 1}']) for k in range(BALLS_DRAWN))

    # --- Method 1: frequency-ranked ------------------------------------------
    print(f"\nTop {len(selections)} frequency-ranked selections for {date_label} "
          f"(higher historical frequency score first):")
    print_selection_table(selections)

    predicted_combination, predicted_id, _ = selections[0]
    if predicted_id in reference:
        print(f"  This exact combination was drawn before: {reference[predicted_id]}")
    else:
        print("  This exact combination has never appeared in the reference results.")
    print(f"  Balls shared with the last draw ({last['Date']}, {last['Combination']}): "
          f"{match_count(predicted_combination, last_combo)}")

    # --- Method 2: mirror of the latest draw ---------------------------------
    print(f"\nTop {len(mirror_selections)} mirror-of-latest-draw selections for {date_label} "
          f"(latest draw {last['Combination']}; 2nd number fixed to the "
          f"{len(mirror_selections)} most drawn balls):")
    print_selection_table(mirror_selections, show_anchor=True)

    # --- Method 3: quick pick within the mirror id range ---------------------
    if quick_selections:
        mirror_ids = [selection[1] for selection in mirror_selections]
        print(f"\n{len(quick_selections)} quick picks for {date_label} "
              f"(random ids between {min(mirror_ids):,} and {max(mirror_ids):,}, "
              f"avoiding {excluded_count:,} previously drawn ids):")
        print_selection_table(quick_selections)

    print(f"\nOdds for any one of these tickets on {date_label}:")
    print(f"  Jackpot (6 of 6):            1 to {lottery_odds():,.0f}")
    print(f"  5 + bonus (remaining pool):  1 to {bonus_remaining_pool_odds():,.0f}")
    print(f"  6 + bonus (separate pool):   1 to {bonus_separate_pool_odds():,.0f}")
    print(f"  3 of 6:                      1 to {lottery_odds(m=3):,.2f}")


def main() -> None:
    """
    Main function for the 6/49 prediction process
    """
    df = load_and_prepare_data(FILE_PATH)
    reference = load_reference(REFERENCE_PATH)

    print_odds_table()

    frequencies = ball_frequencies(df)

    last = df.iloc[-1]
    last_draw = tuple(int(last[f'b{k + 1}']) for k in range(BALLS_DRAWN))
    mirror_selections = mirror_latest_draw(frequencies, last_draw)

    # ids already drawn in either file, plus the mirror picks themselves
    drawn_ids = set(df['Numbers'].astype(int)) | {int(i) for i in reference}
    mirror_ids = [selection[1] for selection in mirror_selections]
    excluded_ids = drawn_ids | set(mirror_ids)

    selections = best_possibility(frequencies,
                                  quick_pick_range=(min(mirror_ids), max(mirror_ids)),
                                  excluded_ids=excluded_ids)
    ranked_selections = selections[:SELECTION_COUNT]
    quick_selections = selections[SELECTION_COUNT:]
    predicted_combination, predicted_id, _ = ranked_selections[0]

    next_date = datetime.strptime(PREDICTED_DATE, '%d-%b-%y')

    print_predictions(df, reference, frequencies, next_date, ranked_selections,
                      mirror_selections, quick_selections, len(drawn_ids))
    plot_results(df, frequencies, next_date, predicted_combination, predicted_id,
                 mirror_selections, quick_selections)


if __name__ == "__main__":
    main()