import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pandas import DataFrame
from datetime import datetime
from math import comb
from itertools import combinations

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
PREDICTED_DATE = "07-Oct-26"
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

# Tickets printed by each strategy
SELECTION_COUNT = 10
POOL_SIZES = (12, 16, 20, 24)   # a strategy's top balls; widened until enough tickets pass
MAX_SHARED_BALLS = 4            # two tickets of one strategy share at most this many balls

# --- Hot number strategies ---------------------------------------------------
RECENT_DRAWS = 17           # "recent frequency" window (typically 10-17 games)
HOT_MAX_GAMES_OUT = 5       # a ball is "hot" when out for this many games or less
TREND_YEAR = None           # year for the yearly trend; None = year of PREDICTED_DATE

# --- Cold number strategies --------------------------------------------------
OVERDUE_DRAWS = 13          # a ball is "overdue" when absent from the last 13 draws
COLD_MONTHS = 8             # window for "infrequent over the last 8 months"
CALENDAR_MAX = 31           # "calendar numbers" are 1..31
MAX_CALENDAR_BALLS = 2      # calendar numbers allowed on an "avoiding clusters" ticket

# --- Complementary statistical tactics (applied to every ticket) -------------
ODD_COUNTS = (2, 3, 4)      # odd balls allowed: 3:3, 2:4 or 4:2 odd/even
LOW_MAX = 25                # low = 1..25, high = 26..49
LOW_COUNTS = (2, 3, 4)      # low balls allowed: 3:3, 2:4 or 4:2 low/high
SUM_RANGE = (115, 185)      # allowed total of the six numbers
SUM_MIDPOINT = 151          # most common sum; used to break ties between tickets
REQUIRE_REPEAT_HIT = False  # True = every ticket must repeat a ball of the latest draw
                            # (overdue balls can never do that, so those tickets relax it)

# --- Quick Pool-and-Filter method ---------------------------------------------
QUICK_POOL_RECENT_DRAWS = 50    # window for the hot bucket
QUICK_POOL_HOT = 5              # most drawn in that window
QUICK_POOL_COLD = 8             # 3 - most overdue (least drawn in the window breaks ties) brings in 37, 28, 7
QUICK_POOL_MID = 11             # 2 - all-time frequency nearest the average brings in 4, 23
QUICK_POOL_REPEAT = 2           # 2 - balls of the most recent draw
QUICK_POOL_TICKETS = 200         # final tickets kept after filtering
QUICK_POOL_SUM_RANGE = (100, 185)   # this method's own sum filter (standard is 115-185);
                                    # lowered so the target's sum of 100 passes
QUICK_POOL_SUM_TARGET = 200     # final tickets are those with sums closest to this
                                # (standard is 152; set to the target's sum)
QUICK_POOL_MAX_SIZE = 30        # largest pool the target fit may build (C(30,6) = 593,775)

# Target combination to reproduce with the Quick Pool-and-Filter method
TARGET_COMBINATION = None   # None = no target | (1, 4, 7, 23, 28, 37) = target combination
QUICK_POOL_FIT_TARGET = False    # True = widen the buckets just enough to put every
                                # target ball in the pool; False = only report on it
QUICK_POOL_CSV = "quick_pool_tickets.csv"   # final tickets are also saved here (None = don't)

BALL_COLUMNS = [f'b{k + 1}' for k in range(BALLS_DRAWN)]


# ---------------------------------------------------------------------------
# Combination <-> id mapping (lexicographic rank, 1-based) - the "FREQ" id
# ---------------------------------------------------------------------------
def combination_to_id(combination, n=BALLS_IN_POOL, r=BALLS_DRAWN) -> int:
    """
    Map a sorted r-combination of 1..n to its 1-based lexicographic rank.
    e.g. (1, 2, 3, 4, 13, 48) -> 359   (matches REFERENCE_PATH)
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
    """Odds (1 to x) of matching exactly m of the r balls drawn from a pool of n."""
    return comb(n, r) / (comb(r, m) * comb(n - r, r - m))


def bonus_remaining_pool_odds(n=BALLS_IN_POOL, r=BALLS_DRAWN, m=BONUS_REMAINING_MATCHES) -> float:
    """
    Bonus ball taken from the remaining pool: odds of matching m main numbers
    AND the bonus ball (default m = r - 1, i.e. "5 + bonus" for a 6/49 game).
    """
    favourable = comb(r, m) * 1 * comb(n - r - 1, r - m - 1)
    return comb(n, r) / favourable


def bonus_separate_pool_odds(n=BALLS_IN_POOL, r=BALLS_DRAWN, m=NUMBER_OF_MATCHES,
                             bonus_drawn=BONUS_POOL_BALLS_DRAWN,
                             bonus_matches=BONUS_POOL_MATCHES,
                             bonus_pool=BONUS_POOL_SIZE) -> float:
    """
    Bonus ball taken from a separate pool: odds of m main matches AND
    bonus_matches of the bonus balls.  The draws are independent, so the odds multiply.
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

    Returns:
        DataFrame: date, id, and one column per ball (b1..b6), oldest draw first
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
# Ball statistics
# ---------------------------------------------------------------------------
def ball_frequencies(df: DataFrame) -> np.ndarray:
    """
    Count how often each ball 1..49 appears in the draws of `df`.

    Returns:
        numpy.array: index 0 is ball 1, ..., index 48 is ball 49
    """
    balls = df[BALL_COLUMNS].to_numpy(dtype=int).ravel()
    return np.bincount(balls, minlength=BALLS_IN_POOL + 1)[1:]


def games_out(df: DataFrame) -> np.ndarray:
    """
    Draws since each ball last appeared: 0 = in the latest draw, 1 = last seen
    the draw before, ...  A ball never drawn gets len(df).
    """
    draws = df[BALL_COLUMNS].to_numpy(dtype=int)
    out = np.full(BALLS_IN_POOL, len(draws), dtype=int)
    for index, draw in enumerate(draws):
        out[draw - 1] = len(draws) - 1 - index
    return out


def match_count(combination_a, combination_b) -> int:
    """Number of balls shared by two combinations."""
    return len(set(combination_a) & set(combination_b))


def rank_balls(*keys) -> list:
    """
    Order balls 1..49 by the given per-ball key arrays (ascending; negate an
    array for "highest first"), ball number breaking the final tie.
    """
    return sorted(range(1, BALLS_IN_POOL + 1),
                  key=lambda ball: tuple(int(key[ball - 1]) for key in keys) + (ball,))


# ---------------------------------------------------------------------------
# Complementary statistical tactics
# ---------------------------------------------------------------------------
def odd_count(combination) -> int:
    """Number of odd balls in a combination."""
    return sum(ball % 2 for ball in combination)


def low_count(combination) -> int:
    """Number of low balls (1..LOW_MAX) in a combination."""
    return sum(ball <= LOW_MAX for ball in combination)


def passes_tactics(combination, last_draw, require_repeat=REQUIRE_REPEAT_HIT) -> bool:
    """
    True when a combination satisfies the complementary tactics:
    odd/even balance, high/low balance, sum range and (optionally) a repeat hit.
    """
    if odd_count(combination) not in ODD_COUNTS:
        return False
    if low_count(combination) not in LOW_COUNTS:
        return False
    if not SUM_RANGE[0] <= sum(combination) <= SUM_RANGE[1]:
        return False
    if require_repeat and match_count(combination, last_draw) == 0:
        return False
    return True


def tactic_statistics(df: DataFrame) -> dict:
    """
    Measure, on the historical draws, the figures the tactics are based on.

    Returns:
        dict: odd / low count distributions (share per count), share of sums in
              SUM_RANGE, mean and median sum, share of draws repeating at least
              one ball of the previous draw, and share of winning numbers that
              were "hot" (out HOT_MAX_GAMES_OUT games or less) when drawn.
    """
    draws = df[BALL_COLUMNS].to_numpy(dtype=int)
    total = len(draws)
    odd = (draws % 2 == 1).sum(axis=1)
    low = (draws <= LOW_MAX).sum(axis=1)
    sums = draws.sum(axis=1)

    repeats = sum(bool(set(a) & set(b)) for a, b in zip(draws, draws[1:]))

    last_seen, hot, counted = {}, 0, 0
    for index, draw in enumerate(draws):
        for ball in draw:
            if ball in last_seen:
                counted += 1
                hot += (index - last_seen[ball] - 1) <= HOT_MAX_GAMES_OUT
        for ball in draw:
            last_seen[ball] = index

    return {
        'draws': total,
        'odd': {k: float((odd == k).mean()) for k in range(BALLS_DRAWN + 1)},
        'low': {k: float((low == k).mean()) for k in range(BALLS_DRAWN + 1)},
        'sum_in_range': float(((sums >= SUM_RANGE[0]) & (sums <= SUM_RANGE[1])).mean()),
        'sum_mean': float(sums.mean()),
        'sum_median': float(np.median(sums)),
        'repeat': repeats / (total - 1) if total > 1 else 0.0,
        'hot_share': hot / counted if counted else 0.0,
    }


# ---------------------------------------------------------------------------
# Strategies - each returns the balls it favours, best first
# ---------------------------------------------------------------------------
def make_strategy(group, name, description, ranked, scores, unit, notes=(),
                  rule=None, pool_sizes=POOL_SIZES) -> dict:
    """Bundle one strategy: ranked balls, per-ball score and ticket rules."""
    return {'group': group, 'name': name, 'description': description,
            'ranked': ranked, 'scores': np.asarray(scores), 'unit': unit,
            'notes': list(notes), 'rule': rule, 'pool_sizes': pool_sizes}


def recent_frequency_strategy(df: DataFrame, out: np.ndarray) -> dict:
    """Hot: balls drawn most often in the last RECENT_DRAWS draws."""
    recent = ball_frequencies(df.tail(RECENT_DRAWS))
    hot_balls = [ball for ball in range(1, BALLS_IN_POOL + 1) if out[ball - 1] <= HOT_MAX_GAMES_OUT]
    notes = [f"Hot balls (out {HOT_MAX_GAMES_OUT} games or less): "
             f"{len(hot_balls)} of {BALLS_IN_POOL} -> {format_combination(hot_balls)}"]
    return make_strategy(
        'Hot', 'Recent Frequency',
        f"most drawn in the last {min(RECENT_DRAWS, len(df))} draws",
        rank_balls(-recent, out), recent, 'x', notes)


def long_term_leaders_strategy(frequencies: np.ndarray, draw_total: int) -> dict:
    """Hot: balls drawn most often over the whole history."""
    order = rank_balls(-frequencies)
    middle = (BALLS_IN_POOL - BALLS_DRAWN) // 2

    def listing(balls):
        return ", ".join(f"{ball} ({frequencies[ball - 1]}x)" for ball in balls)

    notes = [f"Most frequently drawn balls:   {listing(order[:BALLS_DRAWN])}",
             f"Medium frequently drawn balls: {listing(order[middle:middle + BALLS_DRAWN])}",
             f"Least frequently drawn balls:  {listing(order[-BALLS_DRAWN:])}"]
    return make_strategy(
        'Hot', 'Long-Term Leaders', f"all-time frequency over {draw_total:,} draws",
        order, frequencies, 'x', notes)


def year_trend_strategy(df: DataFrame, year: int) -> dict:
    """Hot: balls drawn most often in one calendar year."""
    notes = []
    in_year = df[df['date'].dt.year == year]
    if in_year.empty:
        fallback = int(df['date'].dt.year.max())
        notes.append(f"No draws dated {year} in {FILE_PATH}; using {fallback} instead.")
        year, in_year = fallback, df[df['date'].dt.year == fallback]
    counts = ball_frequencies(in_year)
    return make_strategy(
        'Hot', f'{year} Trends', f"most drawn in the {len(in_year)} draws of {year}",
        rank_balls(-counts), counts, 'x', notes)


def overdue_strategy(df: DataFrame, out: np.ndarray) -> dict:
    """Cold: balls absent longest, backed by low frequency over COLD_MONTHS months."""
    cutoff = df['date'].max() - pd.DateOffset(months=COLD_MONTHS)
    window = df[df['date'] > cutoff]
    cold_counts = ball_frequencies(window)
    overdue = [ball for ball in rank_balls(-out) if out[ball - 1] >= OVERDUE_DRAWS]
    coldest = rank_balls(cold_counts, -out)[:POOL_SIZES[0]]
    notes = [f"Overdue balls (absent from the last {OVERDUE_DRAWS} draws): "
             f"{len(overdue)} -> {format_combination(overdue) or 'none'}",
             f"Least drawn over the last {COLD_MONTHS} months ({len(window)} draws): "
             + ", ".join(f"{ball} ({cold_counts[ball - 1]}x)" for ball in coldest)]
    return make_strategy(
        'Cold', 'Overdue Selection', "longest absent first",
        rank_balls(-out, cold_counts), out, ' out', notes)


def avoiding_clusters_strategy(frequencies: np.ndarray) -> dict:
    """
    Cold: avoid "calendar numbers" (1..CALENDAR_MAX), which many players pick.
    Tickets are built from the balls above CALENDAR_MAX plus the few low balls
    the high/low tactic needs, with at most MAX_CALENDAR_BALLS calendar numbers.
    """
    by_frequency = rank_balls(-frequencies)
    non_calendar = [ball for ball in by_frequency if ball > CALENDAR_MAX]
    low_balls = [ball for ball in by_frequency if ball <= min(LOW_MAX, CALENDAR_MAX)]
    notes = [f"At most {MAX_CALENDAR_BALLS} calendar numbers (1-{CALENDAR_MAX}) per ticket; "
             f"the rest come from {CALENDAR_MAX + 1}-{BALLS_IN_POOL}.",
             "This only lowers the chance of sharing a jackpot - not the chance of winning one."]

    def rule(combination):
        return sum(ball <= CALENDAR_MAX for ball in combination) <= MAX_CALENDAR_BALLS

    base = len(non_calendar)
    return make_strategy(
        'Cold', 'Avoiding Clusters', "few calendar numbers, ranked by all-time frequency",
        non_calendar + low_balls, frequencies, 'x', notes, rule,
        pool_sizes=(base + 6, base + 10))


def build_strategies(df: DataFrame, frequencies: np.ndarray, trend_year: int) -> list:
    """All hot and cold strategies, in report order."""
    out = games_out(df)
    return [
        recent_frequency_strategy(df, out),
        long_term_leaders_strategy(frequencies, len(df)),
        year_trend_strategy(df, trend_year),
        overdue_strategy(df, out),
        avoiding_clusters_strategy(frequencies),
    ]


# ---------------------------------------------------------------------------
# Ticket building - a strategy's balls filtered by the complementary tactics
# ---------------------------------------------------------------------------
def _select_diverse(candidates, drawn_ids, count, selections) -> None:
    """Append best-first candidates to `selections`, keeping tickets distinct."""
    taken = {selection[1] for selection in selections}
    for combination, score in candidates:
        if len(selections) >= count:
            return
        if any(match_count(combination, selected[0]) > MAX_SHARED_BALLS
               for selected in selections):
            continue
        combination_id = combination_to_id(combination)
        if combination_id in drawn_ids or combination_id in taken:
            continue
        selections.append((combination, combination_id, score))
        taken.add(combination_id)


def build_tickets(strategy: dict, last_draw, drawn_ids=(), count=SELECTION_COUNT) -> dict:
    """
    Turn a strategy's ranked balls into `count` tickets.

    Every 6-combination of the strategy's top balls is scored by the sum of the
    strategy's per-ball scores; combinations failing the complementary tactics,
    already drawn before, or too similar to a better ticket are skipped.  The
    pool of top balls is widened (strategy['pool_sizes']) until enough pass.
    Tickets that only fit with the tactics relaxed are reported in 'relaxed'.

    Returns:
        dict: tickets [(combination, id, score)], pool (balls used), relaxed (ids)
    """
    drawn_ids = set(drawn_ids)
    rule, scores = strategy['rule'], strategy['scores']
    require_repeat = REQUIRE_REPEAT_HIT
    tickets, pool, candidates = [], [], []

    for size in strategy['pool_sizes']:
        pool = sorted(strategy['ranked'][:size])
        # repeat hits are impossible when the pool has no ball of the latest draw
        require_repeat = REQUIRE_REPEAT_HIT and bool(set(pool) & set(last_draw))
        candidates = [(combination, int(sum(scores[ball - 1] for ball in combination)))
                      for combination in combinations(pool, BALLS_DRAWN)
                      if rule is None or rule(combination)]
        candidates.sort(key=lambda c: (-c[1], abs(sum(c[0]) - SUM_MIDPOINT), c[0]))
        tickets = []
        _select_diverse([c for c in candidates if passes_tactics(c[0], last_draw, require_repeat)],
                        drawn_ids, count, tickets)
        if len(tickets) == count:
            break

    strict = {ticket[1] for ticket in tickets}
    _select_diverse(candidates, drawn_ids, count, tickets)     # relax the tactics if short
    relaxed = {ticket[1] for ticket in tickets} - strict
    return {'tickets': tickets, 'pool': pool, 'relaxed': relaxed}


# ---------------------------------------------------------------------------
# Quick Pool-and-Filter method
# ---------------------------------------------------------------------------
QUICK_POOL_BUCKETS = ('Hot', 'Cold/Overdue', 'Mid-range', 'Repeat')


def quick_pool_sizes() -> dict:
    """The configured size of every bucket."""
    return dict(zip(QUICK_POOL_BUCKETS, (QUICK_POOL_HOT, QUICK_POOL_COLD,
                                         QUICK_POOL_MID, QUICK_POOL_REPEAT)))


def quick_pool_rankings(df: DataFrame, frequencies: np.ndarray) -> dict:
    """
    Rank the balls for each bucket of the pool, best first:

      Hot          - most drawn in the last QUICK_POOL_RECENT_DRAWS draws
      Cold/Overdue - most overdue (least drawn in that window breaks ties)
      Mid-range    - all-time frequency nearest the average
      Repeat       - balls of the most recent draw (most drawn in the window first)

    Returns:
        dict: bucket name -> (ranked balls, function giving a ball's detail text)
    """
    window = df.tail(QUICK_POOL_RECENT_DRAWS)
    recent = ball_frequencies(window)
    out = games_out(df)
    distance = np.abs(frequencies - frequencies.mean())
    last_draw = set(int(v) for v in df.iloc[-1][BALL_COLUMNS])
    return {
        'Hot': (rank_balls(-recent, out),
                lambda b: f"{recent[b - 1]}x in last {len(window)}"),
        'Cold/Overdue': (rank_balls(-out, recent),
                         lambda b: f"{out[b - 1]} out"),
        'Mid-range': (sorted(range(1, BALLS_IN_POOL + 1), key=lambda b: (distance[b - 1], b)),
                      lambda b: f"{frequencies[b - 1]}x all-time"),
        'Repeat': ([b for b in rank_balls(-recent) if b in last_draw],
                   lambda b: f"latest draw, {recent[b - 1]}x in last {len(window)}"),
    }


def build_quick_pool(rankings: dict, sizes: dict) -> list:
    """
    Step 1 - fill the buckets in order (Hot, Cold/Overdue, Mid-range, Repeat).
    A ball already in the pool is skipped, so the pool always holds distinct balls.

    Returns:
        list: (bucket name, [(ball, detail text), ...]) entries
    """
    used, buckets = set(), []
    for name in QUICK_POOL_BUCKETS:
        ranked, detail = rankings[name]
        picked = [ball for ball in ranked if ball not in used][:sizes[name]]
        used.update(picked)
        buckets.append((name, [(ball, detail(ball)) for ball in picked]))
    return buckets


def _pool_balls(buckets: list) -> set:
    return {ball for _, balls in buckets for ball, _ in balls}


def fit_bucket_sizes(rankings: dict, sizes: dict, target) -> dict:
    """
    Widen the buckets just enough for every ball of `target` to enter the pool.
    Each missing ball goes to the bucket that needs the smallest increase
    (the Repeat bucket only when the ball is in the most recent draw).

    Returns:
        dict: the widened bucket sizes
    """
    sizes = dict(sizes)
    while True:
        missing = sorted(set(target) - _pool_balls(build_quick_pool(rankings, sizes)))
        if not missing:
            return sizes
        ball = missing[0]
        options = []
        for name in QUICK_POOL_BUCKETS:
            if ball not in rankings[name][0]:
                continue
            trial = dict(sizes)
            while ball not in _pool_balls(build_quick_pool(rankings, trial)):
                trial[name] += 1
            options.append((trial[name] - sizes[name], QUICK_POOL_BUCKETS.index(name), trial))
        sizes = min(options)[2]


def quick_pool_and_filter(df: DataFrame, frequencies: np.ndarray, drawn_ids=(),
                          target=TARGET_COMBINATION, fit_target=QUICK_POOL_FIT_TARGET) -> dict:
    """
    Quick Pool-and-Filter method.

      Step 1 - build the pool (see build_quick_pool).  With `fit_target`, the
               buckets are first widened so every ball of `target` is in it.
      Step 2 - keep only "balanced" combinations of the pool: odd/even split in
               ODD_COUNTS, sum in QUICK_POOL_SUM_RANGE, low/high split in
               LOW_COUNTS (at least 2 low and 2 high).
      Step 3 - sort by sum closest to QUICK_POOL_SUM_TARGET and keep the first
               QUICK_POOL_TICKETS, skipping combinations already drawn.

    Returns:
        dict: buckets, pool, sizes (used) and base_sizes (configured), raw
              (combination count), eliminated (per filter, each measured on the
              raw combinations), valid (count after all filters), already_drawn
              (valid ones skipped), tickets [(combination, id, distance of the
              sum from the target)], and target (report on `target`, or None)
    """
    drawn_ids = set(drawn_ids)
    rankings = quick_pool_rankings(df, frequencies)
    base_sizes = quick_pool_sizes()
    sizes, fit_note = base_sizes, None
    if target and fit_target:
        fitted = fit_bucket_sizes(rankings, base_sizes, target)
        fitted_pool = len(_pool_balls(build_quick_pool(rankings, fitted)))
        if fitted_pool <= QUICK_POOL_MAX_SIZE:
            sizes = fitted
        else:
            fit_note = (f"Fitting the target needs a {fitted_pool}-number pool "
                        f"(limit QUICK_POOL_MAX_SIZE = {QUICK_POOL_MAX_SIZE}); buckets left as configured.")

    buckets = build_quick_pool(rankings, sizes)
    pool = sorted(_pool_balls(buckets))

    filters = {
        'Odd/Even': lambda c: odd_count(c) in ODD_COUNTS,
        'Sum range': lambda c: QUICK_POOL_SUM_RANGE[0] <= sum(c) <= QUICK_POOL_SUM_RANGE[1],
        'Low/High mix': lambda c: low_count(c) in LOW_COUNTS,
    }
    raw = 0
    eliminated = dict.fromkeys(filters, 0)
    valid = []
    for combination in combinations(pool, BALLS_DRAWN):
        raw += 1
        passed = True
        for name, keep in filters.items():
            if not keep(combination):
                eliminated[name] += 1
                passed = False
        if passed:
            valid.append(combination)
    valid.sort(key=lambda c: (abs(sum(c) - QUICK_POOL_SUM_TARGET), c))

    target_report = None
    if target:
        target = tuple(sorted(int(v) for v in target))
        target_report = {
            'combination': target, 'id': combination_to_id(target),
            'missing': sorted(set(target) - set(pool)),
            'checks': {name: bool(keep(target)) for name, keep in filters.items()},
            'drawn': combination_to_id(target) in drawn_ids,
            'rank': None, 'ranks': {}, 'bucket': {}, 'fit_note': fit_note,
        }
        for ball in target:
            target_report['ranks'][ball] = {
                name: (rankings[name][0].index(ball) + 1 if ball in rankings[name][0] else None)
                for name in QUICK_POOL_BUCKETS}
            target_report['bucket'][ball] = next(
                (name for name, balls in buckets if ball in [b for b, _ in balls]), None)

    # valid combinations that were drawn before are skipped
    pool_set = set(pool)
    drawn_valid = set()
    for drawn_id in drawn_ids:
        drawn = id_to_combination(drawn_id)
        if pool_set.issuperset(drawn) and all(keep(drawn) for keep in filters.values()):
            drawn_valid.add(drawn)
    already_drawn = len(drawn_valid)

    tickets, rank = [], 0
    want_rank = bool(target_report) and target_report['combination'] in set(valid) \
        and not target_report['drawn']
    for combination in valid:
        if combination in drawn_valid:
            continue
        rank += 1
        if want_rank and combination == target_report['combination']:
            target_report['rank'] = rank
            want_rank = False
        if len(tickets) < QUICK_POOL_TICKETS:
            tickets.append((combination, combination_to_id(combination),
                            abs(sum(combination) - QUICK_POOL_SUM_TARGET)))
        elif not want_rank:
            break
    return {'buckets': buckets, 'pool': pool, 'sizes': sizes, 'base_sizes': base_sizes,
            'raw': raw, 'eliminated': eliminated, 'valid': len(valid),
            'already_drawn': already_drawn, 'tickets': tickets, 'relaxed': set(),
            'target': target_report}


def explain_target(quick: dict) -> None:
    """
    Report on TARGET_COMBINATION: where each of its balls ranks in the buckets,
    whether the pool holds them all, which filters it passes, and where it
    lands in the final sorted list.
    """
    report = quick['target']
    if not report:
        return
    combination = report['combination']
    odd, low = odd_count(combination), low_count(combination)
    print(f"\nTarget combination {format_combination(combination)} (id {report['id']:,}, "
          f"odd:even {odd}:{BALLS_DRAWN - odd}, low:high {low}:{BALLS_DRAWN - low}, "
          f"sum {sum(combination)}):")

    print(f"  {'Ball':<6}{'Hot rank':>10}{'Cold rank':>11}{'Mid rank':>10}{'Repeat rank':>13}   In pool via")
    for ball in combination:
        ranks = report['ranks'][ball]
        cells = [str(ranks[name]) if ranks[name] else '-' for name in QUICK_POOL_BUCKETS]
        print(f"  {ball:<6}{cells[0]:>10}{cells[1]:>11}{cells[2]:>10}{cells[3]:>13}   "
              f"{report['bucket'][ball] or 'NOT IN POOL'}")
    print("  (a ball enters a bucket when its rank is within the bucket size; "
          "Repeat rank '-' = not in the latest draw)")

    if report['fit_note']:
        print(f"  {report['fit_note']}")
    for name, passed in report['checks'].items():
        print(f"  {name + ' filter:':<22}{'pass' if passed else 'FAIL'}")

    if report['missing']:
        fix = ("Raise QUICK_POOL_MAX_SIZE to allow the wider pool." if report['fit_note']
               else "Set QUICK_POOL_FIT_TARGET = True or widen the buckets.")
        print(f"  Result: cannot be produced - {format_combination(report['missing'])} "
              f"not in the pool. {fix}")
    elif not all(report['checks'].values()):
        print("  Result: cannot be produced - it fails the filters marked FAIL above.")
    elif report['drawn']:
        print("  Result: it passes, but was already drawn before, so it is skipped.")
    elif report['rank'] <= QUICK_POOL_TICKETS:
        print(f"  Result: produced - rank {report['rank']} of the final {len(quick['tickets'])} tickets.")
    else:
        hint = f"Raise QUICK_POOL_TICKETS to {report['rank']:,}"
        if QUICK_POOL_SUM_TARGET != sum(combination):
            hint += f" or set QUICK_POOL_SUM_TARGET = {sum(combination)}"
        print(f"  Result: it passes the filters at rank {report['rank']:,}, beyond the cut of "
              f"{QUICK_POOL_TICKETS}. {hint}.")


def save_quick_pool(quick: dict, last_draw, path=QUICK_POOL_CSV) -> None:
    """Save the final Quick Pool-and-Filter tickets as a CSV file."""
    rows = []
    for rank, (combination, combination_id, _) in enumerate(quick['tickets'], start=1):
        odd, low = odd_count(combination), low_count(combination)
        rows.append({'Rank': rank, 'Numbers': format_combination(combination),
                     'Combination id': combination_id,
                     'Odd:Even': f"{odd}:{BALLS_DRAWN - odd}",
                     'Low:High': f"{low}:{BALLS_DRAWN - low}",
                     'Sum': sum(combination),
                     'Repeats': match_count(combination, last_draw)})
    pd.DataFrame(rows).to_csv(path, index=False)


def print_quick_pool(quick: dict, last_draw, date_label: str, repeat_rate: float) -> None:
    """Print the three steps of the Quick Pool-and-Filter method and the target report."""
    print(f"\n{'=' * 100}\nQuick Pool-and-Filter Method for {date_label}\n{'=' * 100}")

    print(f"\nStep 1 - {len(quick['pool'])}-number pool: {format_combination(quick['pool'])}")
    for name, balls in quick['buckets']:
        print(f"  {name:<14}{len(balls):>2}  " + ", ".join(f"{ball} ({detail})" for ball, detail in balls))
    if quick['sizes'] != quick['base_sizes']:
        changes = ", ".join(f"{name} {quick['base_sizes'][name]} -> {quick['sizes'][name]}"
                            for name in QUICK_POOL_BUCKETS
                            if quick['sizes'][name] != quick['base_sizes'][name])
        print(f"  Buckets widened to hold the target combination: {changes}")
    print(f"  ({repeat_rate:.0%} of draws in {FILE_PATH} repeat at least one ball of the previous draw)")

    raw = quick['raw']
    low_rule = f"at least {min(LOW_COUNTS)} low (1-{LOW_MAX}) and {BALLS_DRAWN - max(LOW_COUNTS)} high"
    rules = {'Odd/Even': "/".join(f"{k}:{BALLS_DRAWN - k}" for k in sorted(ODD_COUNTS, reverse=True)) + " splits only",
             'Sum range': f"sums {QUICK_POOL_SUM_RANGE[0]}-{QUICK_POOL_SUM_RANGE[1]} only",
             'Low/High mix': low_rule}
    print(f"\nStep 2 - filter the C({len(quick['pool'])},{BALLS_DRAWN}) = {raw:,} raw combinations:")
    for name, removed in quick['eliminated'].items():
        share = removed / raw if raw else 0.0
        print(f"  {name:<14}{rules[name]:<34} eliminates {removed:>7,} ({share:.0%})")
    print(f"  Balanced combinations left after all three filters: {quick['valid']:,}")
    if quick['already_drawn']:
        print(f"  {quick['already_drawn']} of them were drawn before and are skipped.")

    target_id = quick['target']['id'] if quick['target'] else None
    print(f"\nStep 3 - the {len(quick['tickets'])} with sums closest to {QUICK_POOL_SUM_TARGET}:")
    print_ticket_table(quick, last_draw, f'Off {QUICK_POOL_SUM_TARGET}', expected=0,
                       highlight_id=target_id)
    if len(quick['tickets']) < QUICK_POOL_TICKETS:
        print(f"  Only {len(quick['tickets'])} balanced combinations are available "
              f"from this pool (wanted {QUICK_POOL_TICKETS}).")
    if QUICK_POOL_CSV:
        print(f"  Saved to {QUICK_POOL_CSV}")

    explain_target(quick)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_labels(ax, x, y, color):
    """Add value labels to plot points."""
    for xi, yi in zip(x, y):
        ax.annotate(f'{int(yi):,}', (xi, yi), textcoords="offset points",
                    xytext=(0, 10), ha='center', va='bottom', color=color, fontsize=7)


def plot_results(df: DataFrame, frequencies: np.ndarray, next_date: datetime,
                 strategies: list, results: list, quick: dict = None) -> None:
    """
    Two-panel visualization:
      top    - combination id of every draw over time, plus the top ticket of
               each strategy and of the quick pool at next_date
      bottom - how often each ball 1..49 was drawn; red = hot, blue = overdue
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
    styles = [('red', '*'), ('darkorange', 'D'), ('gold', 'P'), ('blue', 'v'), ('teal', 'X')]
    for (color, marker), strategy, result in zip(styles, strategies, results):
        if not result['tickets']:
            continue
        combination, combination_id, _ = result['tickets'][0]
        ax_ids.plot([next_date], [combination_id], color=color, marker=marker,
                    markersize=10, markeredgecolor='black', linestyle='none',
                    label=f"{strategy['name']}: {combination_id:,} -> {format_combination(combination)}")
    if quick and quick['tickets']:
        combination, combination_id, _ = quick['tickets'][0]
        ax_ids.plot([next_date], [combination_id], color='limegreen', marker='s',
                    markersize=8, markeredgecolor='black', linestyle='none',
                    label=f"Quick Pool-and-Filter: {combination_id:,} -> {format_combination(combination)}")
    plot_labels(ax_ids, df['date'].tail(5), df['Numbers'].tail(5), 'purple')

    ax_ids.set_xlabel('Date')
    ax_ids.set_ylabel('Combination id (1 .. C(49,6))')
    ax_ids.set_title('6/49 draw results as combination ids, with the top ticket of each strategy')
    ax_ids.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=8)
    ax_ids.grid(True)
    ax_ids.tick_params(axis='x', rotation=45)

    # --- Bottom: ball frequency, coloured hot / overdue --------------------
    balls = np.arange(1, BALLS_IN_POOL + 1)
    out = games_out(df)
    colors = ['tomato' if o <= HOT_MAX_GAMES_OUT else
              'steelblue' if o >= OVERDUE_DRAWS else 'lightgrey' for o in out]
    ax_freq.bar(balls, frequencies, color=colors)
    mean_line = ax_freq.axhline(frequencies.mean(), color='red', linestyle='--',
                                label=f'mean {frequencies.mean():.1f}')
    for b, f in zip(balls, frequencies):
        ax_freq.annotate(str(f), (b, f), textcoords="offset points", xytext=(0, 2),
                         ha='center', fontsize=7)
    ax_freq.set_xticks(balls)
    ax_freq.set_xlabel('Ball')
    ax_freq.set_ylabel('Times drawn')
    ax_freq.set_title(f'Ball frequency over {len(df):,} draws')
    ax_freq.legend(handles=[
        Patch(color='tomato', label=f'hot (out {HOT_MAX_GAMES_OUT} games or less)'),
        Patch(color='steelblue', label=f'overdue (absent {OVERDUE_DRAWS}+ draws)'),
        Patch(color='lightgrey', label='neither'),
        mean_line], loc='lower right', fontsize=8)
    ax_freq.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('649_predictions.png', dpi=120)
    plt.show()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_ticket_table(result: dict, last_draw, score_label: str,
                       expected=SELECTION_COUNT, highlight_id=None) -> None:
    """Print a strategy's tickets with their tactic figures."""
    print(f"  {'Rank':<6}{'Numbers':<22}{'Combination id':>15}{score_label:>10}"
          f"{'Odd:Even':>10}{'Low:High':>10}{'Sum':>6}{'Repeats':>9}")
    for rank, (combination, combination_id, score) in enumerate(result['tickets'], start=1):
        odd, low = odd_count(combination), low_count(combination)
        flag = " *" if combination_id in result['relaxed'] else ""
        if highlight_id is not None and combination_id == highlight_id:
            flag += "  <- target"
        print(f"  {rank:<6}{format_combination(combination):<22}{combination_id:>15,}{score:>10,}"
              f"{f'{odd}:{BALLS_DRAWN - odd}':>10}{f'{low}:{BALLS_DRAWN - low}':>10}"
              f"{sum(combination):>6}{match_count(combination, last_draw):>9}{flag}")
    if result['relaxed']:
        print("  * tactics relaxed - not enough combinations of this strategy's balls satisfy them")
    if len(result['tickets']) < expected:
        print(f"  Only {len(result['tickets'])} distinct tickets could be built.")


def print_tactics(stats: dict) -> None:
    """Print the complementary tactics with the figures measured on the data."""
    def ratio(distribution, count):
        return f"{count}:{BALLS_DRAWN - count} {distribution[count]:.0%}"

    def summary(distribution, allowed):
        share = sum(distribution[k] for k in allowed)
        return ", ".join(ratio(distribution, k) for k in sorted(allowed, reverse=True)) + f"  (total {share:.0%})"

    repeat_rule = "required" if REQUIRE_REPEAT_HIT else "shown per ticket, not required"
    print(f"\nComplementary statistical tactics (measured over {stats['draws']:,} draws; "
          f"every ticket below must satisfy them):")
    print(f"  Odd/Even balance:  {summary(stats['odd'], ODD_COUNTS)}")
    print(f"  Low/High balance:  {summary(stats['low'], LOW_COUNTS)}   "
          f"[low 1-{LOW_MAX}, high {LOW_MAX + 1}-{BALLS_IN_POOL}]")
    print(f"  Sum range:         {stats['sum_in_range']:.0%} of draws total "
          f"{SUM_RANGE[0]}-{SUM_RANGE[1]}  (mean {stats['sum_mean']:.1f}, median {stats['sum_median']:.0f})")
    print(f"  Repeat hits:       {stats['repeat']:.0%} of draws repeat at least one ball "
          f"of the previous draw  ({repeat_rule})")
    print(f"  Hot numbers:       {stats['hot_share']:.0%} of winning numbers were out "
          f"{HOT_MAX_GAMES_OUT} games or less when drawn")


def print_predictions(df: DataFrame, reference: dict, next_date: datetime,
                      strategies: list, results: list, stats: dict,
                      quick: dict = None) -> None:
    """
    Print the recent draws, the measured tactics, each hot / cold strategy with
    its pool of balls and its tickets, and the Quick Pool-and-Filter method.
    """
    print("\nLast 10 draws (id -> combination):")
    for _, row in df.tail(10).iterrows():
        in_ref = f"  [in {REFERENCE_PATH}: {reference[row['Numbers']]}]" if row['Numbers'] in reference else ""
        print(f"  {row['Date']:<10} {row['Numbers']:>12,}  ->  {row['Combination']}{in_ref}")

    seen_in_reference = df['Numbers'].isin(reference.keys()).sum() if reference else 0
    print(f"\n{seen_in_reference} of {len(df)} result ids also appear in {REFERENCE_PATH}.")

    last = df.iloc[-1]
    last_draw = tuple(int(last[column]) for column in BALL_COLUMNS)
    date_label = next_date.strftime('%d-%b-%y')

    print_tactics(stats)

    group = None
    for strategy, result in zip(strategies, results):
        if strategy['group'] != group:
            group = strategy['group']
            print(f"\n{'=' * 100}\n{group} Number Strategies for {date_label} "
                  f"(latest draw {last['Date']}: {last['Combination']})\n{'=' * 100}")
        scores, unit = strategy['scores'], strategy['unit']
        ordered_pool = [ball for ball in strategy['ranked'] if ball in set(result['pool'])]
        print(f"\n- {strategy['name']}: {strategy['description']}")
        print(f"  Pool ({len(ordered_pool)} balls): "
              + ", ".join(f"{ball} ({scores[ball - 1]}{unit})" for ball in ordered_pool))
        for note in strategy['notes']:
            print(f"  {note}")
        print_ticket_table(result, last_draw, 'Score')

    if quick:
        print_quick_pool(quick, last_draw, date_label, stats['repeat'])

    print(f"\nOdds for any one of these tickets on {date_label} (the same for every combination):")
    print(f"  Jackpot (6 of 6):            1 to {lottery_odds():,.0f}")
    print(f"  5 + bonus (remaining pool):  1 to {bonus_remaining_pool_odds():,.0f}")
    print(f"  6 + bonus (separate pool):   1 to {bonus_separate_pool_odds():,.0f}")
    print(f"  3 of 6:                      1 to {lottery_odds(m=3):,.2f}")


def main() -> None:
    """Main function for the 6/49 strategy report."""
    df = load_and_prepare_data(FILE_PATH)
    reference = load_reference(REFERENCE_PATH)

    print_odds_table()

    next_date = datetime.strptime(PREDICTED_DATE, '%d-%b-%y')
    frequencies = ball_frequencies(df)
    last = df.iloc[-1]
    last_draw = tuple(int(last[column]) for column in BALL_COLUMNS)

    # ids already drawn in either file are never suggested again
    drawn_ids = set(df['Numbers'].astype(int)) | {int(i) for i in reference}

    strategies = build_strategies(df, frequencies, TREND_YEAR or next_date.year)
    results = [build_tickets(strategy, last_draw, drawn_ids) for strategy in strategies]
    stats = tactic_statistics(df)

    quick = quick_pool_and_filter(df, frequencies, drawn_ids)
    if QUICK_POOL_CSV:
        save_quick_pool(quick, last_draw)

    print_predictions(df, reference, next_date, strategies, results, stats, quick)
    plot_results(df, frequencies, next_date, strategies, results, quick)


if __name__ == "__main__":
    main()