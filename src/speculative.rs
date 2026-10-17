//! Position and acceptance rules shared by speculative model backends/drivers.
//! Sampling processors and streaming/stop handling remain engine responsibilities.

use anyhow::{Result, ensure};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundPlan {
    target_start: usize,
    draft_tokens: usize,
    verify_rows: usize,
}

impl RoundPlan {
    pub fn target_start(self) -> usize {
        self.target_start
    }
    pub fn draft_tokens(self) -> usize {
        self.draft_tokens
    }
    pub fn verify_rows(self) -> usize {
        self.verify_rows
    }
    /// remaining_output excludes the already emitted, unconsumed anchor.
    /// None means the pending anchor must remain unconsumed at this boundary.
    pub fn new(
        target_start: usize,
        context: usize,
        remaining_output: usize,
        max_draft: usize,
    ) -> Result<Option<Self>> {
        ensure!(target_start <= context, "target position exceeds context");
        let rows = max_draft
            .saturating_add(1)
            .min(context - target_start)
            .min(remaining_output);
        Ok((rows > 0).then(|| Self {
            target_start,
            draft_tokens: rows - 1,
            verify_rows: rows,
        }))
    }

    /// This is a logical row count, independent of NPU's physical M bucket.
    pub fn consumed_rows(self, accepted_drafts: usize) -> Result<usize> {
        ensure!(
            accepted_drafts <= self.draft_tokens,
            "accepted count exceeds draft window"
        );
        Ok(1 + accepted_drafts)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct GreedyDecision {
    pub accepted_drafts: usize,
    pub consumed_rows: usize,
    /// Accepted proposals followed by the correction/bonus pending anchor.
    /// The old anchor is never emitted a second time.
    pub output: Vec<u32>,
}

/// target_choices[row] is the target argmax after applying processors with the
/// history ending at that input row. This function never mutates sampler history.
pub fn greedy_decision(
    plan: RoundPlan,
    proposals: &[u32],
    target_choices: &[u32],
) -> Result<GreedyDecision> {
    ensure!(
        proposals.len() == plan.draft_tokens && target_choices.len() == plan.verify_rows,
        "speculative validation row mismatch"
    );
    let accepted = proposals
        .iter()
        .zip(target_choices)
        .take_while(|(a, b)| a == b)
        .count();
    let mut output = proposals[..accepted].to_vec();
    output.push(target_choices[accepted]);
    Ok(GreedyDecision {
        accepted_drafts: accepted,
        consumed_rows: plan.consumed_rows(accepted)?,
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tiny history-dependent target oracle exercises repeated partial rejection,
    // emitted vs consumed positions, and full-prefix recomputation.
    fn next(history: &[u32]) -> u32 {
        (history.iter().fold(7, |a, b| (a * 3 + b) % 11) + history.len() as u32) % 11
    }

    #[test]
    fn pending_anchor_matches_autoregressive_oracle_at_every_acceptance_boundary() {
        for forced_accept in 0..=4 {
            let prompt = vec![1, 3, 2];
            let mut target = prompt.clone();
            let mut output = vec![next(&target)];
            let budget = 39;
            while let Some(plan) =
                RoundPlan::new(target.len(), 128, budget - output.len(), 4).unwrap()
            {
                let anchor = *output.last().unwrap();
                let mut temporary = target.clone();
                temporary.push(anchor);
                let mut proposals = Vec::new();
                let mut choices = Vec::new();
                for j in 0..plan.draft_tokens {
                    let expected = next(&temporary);
                    let proposal = if j == forced_accept {
                        (expected + 1) % 11
                    } else {
                        expected
                    };
                    choices.push(expected);
                    proposals.push(proposal);
                    temporary.push(proposal);
                }
                choices.push(next(&temporary));
                let decision = greedy_decision(plan, &proposals, &choices).unwrap();
                assert_eq!(
                    decision.accepted_drafts,
                    forced_accept.min(plan.draft_tokens)
                );
                target.extend_from_slice(
                    &temporary[plan.target_start..plan.target_start + decision.consumed_rows],
                );
                output.extend_from_slice(&decision.output);
                assert_eq!(target.len(), prompt.len() + output.len() - 1);
                assert_eq!(&target[prompt.len()..], &output[..output.len() - 1]);
            }
            let mut expected = prompt.clone();
            for _ in 0..budget {
                expected.push(next(&expected));
            }
            assert_eq!(output, expected[prompt.len()..]);
        }
    }

    #[test]
    fn output_and_context_budgets_include_anchor_input_and_bonus_output() {
        assert_eq!(RoundPlan::new(4, 16, 0, 7).unwrap(), None);
        assert_eq!(RoundPlan::new(16, 16, 9, 7).unwrap(), None);
        for (position, context, remaining) in [(4, 5, 10), (4, 16, 1)] {
            let plan = RoundPlan::new(position, context, remaining, 7)
                .unwrap()
                .unwrap();
            assert_eq!(plan.verify_rows, 1);
            assert_eq!(greedy_decision(plan, &[], &[9]).unwrap().output, [9]);
        }
        for max_draft in [1, 2, 3, 4, 7] {
            let plan = RoundPlan::new(12, 128, 64, max_draft).unwrap().unwrap();
            assert_eq!(plan.verify_rows, max_draft + 1);
            assert_eq!(plan.consumed_rows(0).unwrap(), 1);
        }
    }
}
