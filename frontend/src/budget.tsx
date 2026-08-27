import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

/**
 * The alert budget.
 *
 * A budget is a capacity decision: how much of each scored batch an analyst
 * team can actually work. Argus applies it as an exact top-k by rank within
 * each batch — never as a stored probability cutoff, because Elliptic's score
 * distribution shifts hard in the later time steps and a frozen cutoff
 * collapses recall.
 *
 * 1% is canonical: every metric in the model report is measured at it,
 * and it is the default here and in the backend. The other options exist so
 * the cost of that choice can be seen rather than argued about.
 *
 * The chosen value survives navigating to a case and back, and a reload.
 * Moving the control previews a budget against the stored scores and writes
 * nothing; applying it re-runs the selection and rebuilds the queue, which is
 * a background job with a duration.
 */

export const BUDGETS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05];

export const DEFAULT_BUDGET = 0.01;

export function budgetLabel(value: number): string {
  const percent = value * 100;
  return `${Number.isInteger(percent) ? percent : percent.toFixed(1)}%`;
}

const STORAGE_KEY = "argus.alert-budget";

function readStored(): number {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const value = raw === null ? Number.NaN : Number(raw);
    // Only a value the control can actually represent. Anything else — a hand
    // edited entry, a value from an older build — falls back to canonical.
    return BUDGETS.includes(value) ? value : DEFAULT_BUDGET;
  } catch {
    return DEFAULT_BUDGET;
  }
}

type BudgetState = { budget: number; setBudget: (value: number) => void };

const BudgetContext = createContext<BudgetState>({
  budget: DEFAULT_BUDGET,
  setBudget: () => {},
});

export function BudgetProvider({ children }: { children: ReactNode }) {
  const [budget, setState] = useState(readStored);

  const setBudget = useCallback((value: number) => {
    setState(value);
    try {
      window.localStorage.setItem(STORAGE_KEY, String(value));
    } catch {
      // Private browsing, or storage disabled. The choice still holds for
      // this session; only its survival across reloads is lost.
    }
  }, []);

  const value = useMemo(() => ({ budget, setBudget }), [budget, setBudget]);
  return <BudgetContext.Provider value={value}>{children}</BudgetContext.Provider>;
}

export function useBudget(): BudgetState {
  return useContext(BudgetContext);
}
