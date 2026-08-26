/**
 * The alert budget.
 *
 * A budget is a capacity decision: how much of each scored batch an analyst
 * team can actually work. Argus applies it as an exact top-k by rank within
 * each batch — never as a stored probability cutoff, because Elliptic's score
 * distribution shifts hard in the later time steps and a frozen cutoff
 * collapses recall.
 *
 * 1% is canonical: every metric in `docs/modeling.md` is measured at it,
 * and it is the default here and in the backend. The other options exist so
 * the cost of that choice can be seen rather than argued about.
 */

export const BUDGETS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05];

export const DEFAULT_BUDGET = 0.01;

export function budgetLabel(value: number): string {
  const percent = value * 100;
  return `${Number.isInteger(percent) ? percent : percent.toFixed(1)}%`;
}
