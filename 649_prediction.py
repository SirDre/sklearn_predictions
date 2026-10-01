import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from datetime import datetime
from math import comb

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
PREDICTED_DATE = "30-Sep-26"
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


def best_possibility(frequencies: np.ndarray, r=BALLS_DRAWN) -> tuple:
    """
    Pick the r most frequently drawn balls as the "best possibility" combination
    and map it to its combination id.

    Returns:
        tuple: (combination tuple, combination id)
    """
    order = np.argsort(-frequencies, kind='stable')        # highest count first
    combination = tuple(sorted(int(i) + 1 for i in order[:r]))
    return combination, combination_to_id(combination)


def match_count(combination_a, combination_b) -> int:
    """Number of balls shared by two combinations."""
    return len(set(combination_a) & set(combination_b))


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
                 predicted_combination: tuple, predicted_id: int) -> None:
    """
    Two-panel visualization:
      top    - combination id of every draw over time + predicted id at next_date
      bottom - how often each ball 1..49 was drawn, predicted balls highlighted

    Parameters:
        df (DataFrame): prepared results
        frequencies (numpy.array): draw count per ball
        next_date (datetime): future prediction date
        predicted_combination (tuple): best-possibility combination
        predicted_id (int): its combination id
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
def print_predictions(df: DataFrame, reference: dict, frequencies: np.ndarray,
                      next_date: datetime, predicted_combination: tuple, predicted_id: int) -> None:
    """
    Print the id -> combination mapping of recent draws, the best possibility
    combination, its id and its odds.
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

    print(f"\nBest possibility for {next_date.strftime('%d-%b-%y')}: "
          f"{format_combination(predicted_combination)}  (combination id {predicted_id:,})")
    if predicted_id in reference:
        print(f"  This exact combination was drawn before: {reference[predicted_id]}")
    else:
        print("  This exact combination has never appeared in the reference results.")

    last = df.iloc[-1]
    last_combo = tuple(int(last[f'b{k + 1}']) for k in range(BALLS_DRAWN))
    print(f"  Balls shared with the last draw ({last['Date']}, {last['Combination']}): "
          f"{match_count(predicted_combination, last_combo)}")

    print(f"\nOdds for this (or any) ticket on {next_date.strftime('%d-%b-%y')}:")
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
    predicted_combination, predicted_id = best_possibility(frequencies)

    next_date = datetime.strptime(PREDICTED_DATE, '%d-%b-%y')

    print_predictions(df, reference, frequencies, next_date, predicted_combination, predicted_id)
    plot_results(df, frequencies, next_date, predicted_combination, predicted_id)


if __name__ == "__main__":
    main()