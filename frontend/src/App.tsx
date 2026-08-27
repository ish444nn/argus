import { Navigate, Route, Routes } from "react-router-dom";
import { BudgetProvider } from "./budget";
import { AppShell } from "./components/AppShell";
import { Case } from "./routes/Case";
import { Overview } from "./routes/Overview";
import { Queue } from "./routes/Queue";

export default function App() {
  return (
    // The alert budget is chosen on one screen but is a property of the
    // session, so it sits above the router rather than inside the screen that
    // happens to expose the control.
    <BudgetProvider>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<Overview />} />
          <Route path="queue" element={<Queue />} />
          <Route path="cases/:caseId" element={<Case />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BudgetProvider>
  );
}
