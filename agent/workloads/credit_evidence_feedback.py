"""Opt-in proposal-dependent verification; no phase or access order commands.

The verifier checks evidence presence, not financial correctness. This is a
declared local business fixture extension, not a production underwriting rule.
"""
import hashlib
import json

from .credit_portfolio import PortfolioCreditCursor, PORTFOLIO_INSTRUCTIONS


CONTRACT = 'proposal-evidence-v1'
EVIDENCE_FEEDBACK_INSTRUCTIONS = PORTFOLIO_INSTRUCTIONS + """
The local verifier additionally checks evidence coverage for the proposed action.
For a high-risk company, approving a limit increase requires collateral and
covenant evidence. For any company, increasing the limit by at least 50000
requires sector_outlook evidence. Other proposals need no additional coverage.
The verifier may return verified=false, status=needs_evidence, missing_sources.
That is actionable feedback, not a receipt. You may gather evidence and revise
the proposal, or choose a different justified proposal. Reading ordinary evidence
invalidates the current proposal as usual. Choose query timing and order yourself.
This simulated verifier checks presence of observed sources, not credit quality.
"""


def verification_result(request):
    """Validate the opt-in request and return a receipt or explicit feedback."""
    if request.get('evidence_contract') != CONTRACT:
        raise ValueError('unknown evidence verification contract')
    risk = request.get('risk')
    decision = request.get('decision')
    limit, old = request.get('new_limit'), request.get('observed_limit')
    score = request.get('score')
    observed = request.get('observed_evidence')
    if (risk not in ('low', 'medium', 'high') or decision not in ('approve', 'review', 'deny')
            or type(limit) is not int or not 0 <= limit <= 500000
            or type(old) is not int or not 0 <= old <= 500000
            or type(score) is not int or not 0 <= score <= 100
            or not isinstance(observed, dict)
            or any(k not in ('sector_outlook', 'region_regulation', 'risk_policy',
                'covenant', 'collateral', 'watchlist', 'growth') or not isinstance(v, str)
                for k, v in observed.items())):
        raise ValueError('invalid evidence verification request')
    required = set()
    if risk == 'high' and decision == 'approve' and limit > old:
        required.update(('collateral', 'covenant'))
    if limit-old >= 50000:
        required.add('sector_outlook')
    missing = sorted(required-set(observed))
    if missing:
        return dict(verified=False, status='needs_evidence', missing_sources=missing,
                    service='local-test-verifier', evidence_contract=CONTRACT)
    canonical = json.dumps(request, sort_keys=True).encode('utf-8')
    return dict(verified=True, receipt_id=hashlib.sha256(canonical).hexdigest()[:16],
                service='local-test-verifier', evidence_contract=CONTRACT)


def is_evidence_feedback(response):
    if not isinstance(response, dict):
        return False
    missing = response.get('missing_sources')
    return (response.get('verified') is False and response.get('status') == 'needs_evidence'
            and response.get('evidence_contract') == CONTRACT and not response.get('receipt_id')
            and isinstance(missing, list) and bool(missing)
            and all(isinstance(x, str) and x in ('collateral', 'covenant', 'sector_outlook')
                    for x in missing) and len(set(missing)) == len(missing))


class EvidenceFeedbackPortfolioCursor(PortfolioCreditCursor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verification_feedback = None

    def verification_request(self, risk):
        request = super().verification_request(risk)
        request.update(evidence_contract=CONTRACT, observed_limit=int(self.company['limit']),
                       observed_evidence=dict(self.evidence))
        # A failed or malformed new verification cannot leave an old receipt usable.
        self.receipt = None
        return request

    def accept_verification_response(self, response):
        if is_evidence_feedback(response):
            self.receipt = None
            self.verification_feedback = response
            return response
        result = super().accept_verification_response(response)
        self.verification_feedback = None
        return result

    def planner_context(self, turns_remaining):
        state = json.loads(super().planner_context(turns_remaining))
        if self.verification_feedback is not None:
            state['verification_feedback'] = self.verification_feedback
        return json.dumps(state, sort_keys=True)
