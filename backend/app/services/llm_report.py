from typing import Dict, Any

class ClinicalReportService:
    def generate_summary(self, inference_data: Dict[str, Any]) -> str:
        anomaly = inference_data.get('anomaly', {})
        ood = inference_data.get('ood', {})
        uncertainty = inference_data.get('uncertainty', {})
        abstention = inference_data.get('abstention', {})

        level = anomaly.get('level', 'unknown')
        score = anomaly.get('score', 0.0)
        ood_status = ood.get('status', 'unknown')
        unc_level = uncertainty.get('level', 'unknown')

        if abstention.get('abstain'):
            reason = abstention.get('reason', 'Unknown reason')
            if 'Train or load' in reason:
                return 'ANALYSIS UNAVAILABLE - Train or load a validated ANEVYX model before analysis.'
            if 'ABSTAINED' in reason: 
                return reason
            return f'ANALYSIS UNAVAILABLE - {reason}'

        parts = []

        if level in ['CONSISTENT WITH NORMALITY', 'INCONSISTENT WITH LEARNED NORMALITY']:
            parts.append(f'Reconstruction deviation score ({score:.3f}) ' + 
                         ('falls within the calibrated normative range.' if level == 'CONSISTENT WITH NORMALITY' else 'exceeds the calibrated normative threshold, indicating structural differences from the learned normal distribution.'))
        elif level == 'UNCALIBRATED':
            parts.append('Anomaly threshold has not been calibrated. Score cannot be interpreted.')
        elif level == 'UNAVAILABLE':
            parts.append('Anomaly scoring is unavailable for this scan.')

        if ood_status == 'within_validated_range':
            parts.append('The input is within the model\'s validated distribution.')
        elif ood_status == 'review_recommended':
            parts.append('The input is near the boundary of the training distribution - interpret with caution.')
        elif ood_status == 'out_of_distribution':
            parts.append('The input falls outside the training distribution. Evidence reliability is reduced.')

        if unc_level == 'stable':
            parts.append('Model predictions are stable across test-time augmentations.')
        elif unc_level == 'moderate':
            parts.append('Moderate prediction variance detected.')
        elif unc_level in ('high', 'highly_uncertain'):
            parts.append('High prediction variance - evidence should be interpreted cautiously.')

        return ' '.join(parts)

