import pandas as pd
import numpy as np

class SmartAlertSystem:
    """Intelligent alert system for climate communication monitoring"""
    
    def __init__(self, df):
        self.df = df
        self.baseline_metrics = self.calculate_baseline_metrics()
    
    def calculate_baseline_metrics(self):
        """Calculate baseline metrics for comparison"""
        return {
            'avg_sentiment': self.df['vader_compound'].mean(),
            'avg_engagement': self.df['engagement_score'].mean(),
            'avg_myths_per_day': self.df['myths_detected'].sum() / max(1, self.df['date'].nunique()) if 'date' in self.df.columns else 0,
            'positive_sentiment_rate': (self.df['sentiment_category'] == 'Positive').mean()
        }
    
    def check_sentiment_alerts(self):
        """Check for sentiment-related alerts"""
        alerts = []
        
        # Recent vs historical sentiment (simulate recent data)
        recent_sentiment = self.df.tail(50)['vader_compound'].mean()  # Last 50 comments as "recent"
        historical_sentiment = self.baseline_metrics['avg_sentiment']
        
        sentiment_drop = historical_sentiment - recent_sentiment
        
        if sentiment_drop > 0.3:
            alerts.append({
                'type': 'SENTIMENT_CRASH',
                'severity': 'HIGH',
                'title': 'Significant Sentiment Drop Detected',
                'message': f'Sentiment dropped by {sentiment_drop:.2f} points from baseline',
                'action': 'Review recent content and consider positive follow-up posts',
                'data': {
                    'current': recent_sentiment,
                    'baseline': historical_sentiment,
                    'drop': sentiment_drop
                }
            })
        elif sentiment_drop > 0.15:
            alerts.append({
                'type': 'SENTIMENT_DECLINE',
                'severity': 'MEDIUM',
                'title': 'Moderate Sentiment Decline',
                'message': f'Sentiment down {sentiment_drop:.2f} points - monitoring recommended',
                'action': 'Monitor closely and prepare positive content',
                'data': {
                    'current': recent_sentiment,
                    'baseline': historical_sentiment,
                    'drop': sentiment_drop
                }
            })
        
        return alerts
    
    def check_myth_alerts(self):
        """Check for misinformation-related alerts"""
        alerts = []
        
        # High myth concentration
        high_myth_comments = self.df[self.df['myths_detected'] >= 2]
        
        if len(high_myth_comments) > 3:
            alerts.append({
                'type': 'MISINFORMATION_SURGE',
                'severity': 'HIGH',
                'title': 'Multiple Climate Myths Detected',
                'message': f'{len(high_myth_comments)} comments contain multiple myths',
                'action': 'Deploy fact-checking responses and educational content',
                'data': {
                    'affected_comments': len(high_myth_comments),
                    'total_myths': high_myth_comments['myths_detected'].sum()
                }
            })
        
        # High priority myths
        high_priority_myths = self.df[self.df['high_priority_myth']].shape[0]
        
        if high_priority_myths > 2:
            alerts.append({
                'type': 'HIGH_PRIORITY_MYTHS',
                'severity': 'MEDIUM',
                'title': 'High-Priority Myths Require Response',
                'message': f'{high_priority_myths} comments contain serious misinformation',
                'action': 'Prioritize educational responses with NASA sources',
                'data': {
                    'count': high_priority_myths
                }
            })
        
        return alerts
    
    def check_engagement_alerts(self):
        """Check for engagement-related alerts"""
        alerts = []
        
        # Low engagement posts
        low_engagement = self.df[self.df['engagement_score'] == 0]
        low_engagement_rate = len(low_engagement) / len(self.df)
        
        if low_engagement_rate > 0.6:
            alerts.append({
                'type': 'LOW_ENGAGEMENT',
                'severity': 'MEDIUM',
                'title': 'High Rate of Zero-Engagement Posts',
                'message': f'{low_engagement_rate*100:.1f}% of posts have no engagement',
                'action': 'Review content strategy and posting times',
                'data': {
                    'rate': low_engagement_rate,
                    'count': len(low_engagement)
                }
            })
        
        # High engagement opportunity
        high_engagement = self.df[self.df['engagement_score'] > self.df['engagement_score'].quantile(0.9)]
        
        if len(high_engagement) > 0:
            alerts.append({
                'type': 'HIGH_ENGAGEMENT_OPPORTUNITY',
                'severity': 'POSITIVE',
                'title': 'High-Performing Content Identified',
                'message': f'{len(high_engagement)} posts show exceptional engagement',
                'action': 'Analyze successful patterns and create similar content',
                'data': {
                    'count': len(high_engagement),
                    'avg_engagement': high_engagement['engagement_score'].mean()
                }
            })
        
        return alerts
    
    def check_educational_alerts(self):
        """Check for educational effectiveness alerts"""
        alerts = []
        
        # Questions without responses (if you have this data)
        if 'asks_question' in self.df.columns:
            unanswered_questions = self.df[self.df['asks_question'] & (self.df['commentsCount'] == 0)]
            
            if len(unanswered_questions) > 3:
                alerts.append({
                    'type': 'UNANSWERED_QUESTIONS',
                    'severity': 'MEDIUM',
                    'title': 'Educational Questions Need Attention',
                    'message': f'{len(unanswered_questions)} scientific questions lack responses',
                    'action': 'Respond to educational inquiries to boost learning',
                    'data': {
                        'count': len(unanswered_questions)
                    }
                })
        
        return alerts
    
    def get_all_alerts(self):
        """Get all current alerts"""
        all_alerts = []
        
        all_alerts.extend(self.check_sentiment_alerts())
        all_alerts.extend(self.check_myth_alerts())
        all_alerts.extend(self.check_engagement_alerts())
        all_alerts.extend(self.check_educational_alerts())
        
        # Sort by severity
        severity_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'POSITIVE': 0}
        all_alerts.sort(key=lambda x: severity_order.get(x['severity'], 0), reverse=True)
        
        return all_alerts