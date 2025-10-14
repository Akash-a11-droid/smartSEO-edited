import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
from collections import Counter

# Try to import ML modules, but have fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    ML_AVAILABLE = True
    print("✅ ML modules loaded successfully")
except ImportError as e:
    print(f"⚠️ ML modules not available: {e}")
    ML_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="SmartSEO Analyzer",
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

class HybridSEOAnalyzer:
    def __init__(self):
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'as', 'is', 'was', 'were', 'be', 'been',
            'this', 'that', 'these', 'those', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must'
        }
        
        self.intent_keywords = {
            'transactional': ['buy', 'purchase', 'order', 'price', 'cost', 'deal', 'offer', 'discount', 'sale', 'shop'],
            'informational': ['how', 'what', 'why', 'when', 'where', 'guide', 'tutorial', 'explain', 'learn', 'step'],
            'navigational': ['contact', 'about', 'find', 'location', 'address', 'phone', 'email', 'map', 'support']
        }
        
        self.ml_models = {}
        if ML_AVAILABLE:
            self._initialize_ml_models()

    def _initialize_ml_models(self):
        """Initialize ML models if available"""
        try:
            # Simple TF-IDF and Naive Bayes for intent classification
            self.ml_models['tfidf'] = TfidfVectorizer(
                ngram_range=(1, 2), 
                max_features=1000,
                stop_words='english'
            )
            self.ml_models['classifier'] = MultinomialNB()
            print("✅ ML models initialized")
        except Exception as e:
            print(f"⚠️ ML model initialization failed: {e}")

    def extract_keywords(self, text, top_k=5):
        """Enhanced keyword extraction with TF-IDF if available"""
        if ML_AVAILABLE and self.ml_models.get('tfidf'):
            try:
                # Use TF-IDF for keyword extraction
                tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
                tfidf_matrix = tfidf.fit_transform([text])
                feature_names = tfidf.get_feature_names_out()
                scores = tfidf_matrix.toarray()[0]
                
                # Get top keywords by TF-IDF score
                keyword_scores = list(zip(feature_names, scores))
                keyword_scores.sort(key=lambda x: x[1], reverse=True)
                ml_keywords = [kw for kw, score in keyword_scores[:top_k] if score > 0]
                
                if ml_keywords:
                    return ml_keywords
            except Exception as e:
                print(f"TF-IDF keyword extraction failed: {e}")
        
        # Fallback to frequency-based extraction
        return self._fallback_keyword_extraction(text, top_k)

    def _fallback_keyword_extraction(self, text, top_k=5):
        """Frequency-based keyword extraction fallback"""
        words = re.findall(r'\b[a-z]{3,15}\b', text.lower())
        words = [w for w in words if w not in self.stopwords]
        word_freq = Counter(words)
        return [word for word, count in word_freq.most_common(top_k)]

    def classify_intent(self, text):
        """Hybrid intent classification"""
        if ML_AVAILABLE and self.ml_models.get('classifier'):
            try:
                # This would require training data - using fallback for now
                pass
            except Exception as e:
                print(f"ML intent classification failed: {e}")
        
        # Use rule-based classification
        return self._rule_based_intent_classification(text)

    def _rule_based_intent_classification(self, text):
        """Rule-based intent classification"""
        text_lower = text.lower()
        
        scores = {}
        for intent, keywords in self.intent_keywords.items():
            scores[intent] = sum(1 for kw in keywords if kw in text_lower)
        
        max_score = max(scores.values())
        if max_score == 0:
            return 'informational'
        
        for intent, score in scores.items():
            if score == max_score:
                return intent

    def predict_ranking(self, text):
        """Enhanced ranking prediction using multiple features"""
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        paragraph_count = text.count('\n\n') + 1
        
        # Calculate multiple quality metrics
        keyword_density = self._calculate_keyword_density(text)
        readability_score = self._calculate_readability(text)
        structure_score = self._calculate_structure_score(text)
        
        # Combined scoring formula
        score = min(10, max(1, 
            (word_count / 150) +
            (sentence_count / 8) +
            (paragraph_count / 3) +
            (keyword_density * 10) +
            (readability_score * 2) +
            structure_score
        ))
        
        return round(score, 1)

    def _calculate_keyword_density(self, text):
        """Calculate keyword density score"""
        keywords = self.extract_keywords(text, top_k=3)
        if not keywords:
            return 0.5
        
        text_lower = text.lower()
        density = sum(text_lower.count(kw.lower()) for kw in keywords) / len(text.split())
        return min(1.0, density * 10)

    def _calculate_readability(self, text):
        """Calculate readability score"""
        sentences = re.split(r'[.!?]+', text)
        if not sentences:
            return 0.5
        
        words_per_sentence = [len(sent.split()) for sent in sentences if sent.strip()]
        if not words_per_sentence:
            return 0.5
            
        avg_sentence_length = np.mean(words_per_sentence)
        
        # Ideal sentence length is 15-25 words
        if 15 <= avg_sentence_length <= 25:
            return 1.0
        elif 10 <= avg_sentence_length <= 30:
            return 0.7
        else:
            return 0.4

    def _calculate_structure_score(self, text):
        """Calculate content structure score"""
        score = 0.0
        
        # Check for headings
        if re.search(r'#+|^[A-Z][A-Z\s]{10,}:', text, re.MULTILINE):
            score += 0.3
        
        # Check for paragraphs
        if text.count('\n\n') >= 2:
            score += 0.3
        
        # Check for lists
        if re.search(r'[•\-*]\s|\d+\.', text):
            score += 0.2
        
        # Check for questions (engagement)
        if text.count('?') >= 1:
            score += 0.2
            
        return score

    def suggest_improvements(self, text, intent, keywords):
        """Generate comprehensive SEO improvement suggestions"""
        suggestions = []
        word_count = len(text.split())
        
        # Content length analysis
        if word_count < 300:
            suggestions.append(f"📏 **Expand Content**: Currently {word_count} words. Aim for 800+ words for comprehensive coverage.")
        elif word_count < 500:
            suggestions.append(f"📏 **Good Start**: {word_count} words. Consider expanding to 800+ words for better depth.")
        
        # Keyword optimization
        if keywords:
            # Check if keywords are properly used
            text_lower = text.lower()
            underused_keywords = [kw for kw in keywords if text_lower.count(kw.lower()) < 2]
            if underused_keywords:
                suggestions.append(f"🔑 **Keyword Usage**: Use these keywords more: {', '.join(underused_keywords[:3])}")
        
        # Structure analysis
        structure_score = self._calculate_structure_score(text)
        if structure_score < 0.5:
            suggestions.append("📐 **Improve Structure**: Add headings, bullet points, and paragraph breaks for better organization.")
        
        # Readability improvements
        readability_score = self._calculate_readability(text)
        if readability_score < 0.6:
            suggestions.append("📖 **Enhance Readability**: Vary sentence length and break complex sentences.")
        
        # Intent-specific optimizations
        if intent == 'informational':
            suggestions.extend([
                "🎯 **Informational Optimization**: Add step-by-step guides, examples, and FAQs.",
                "💡 **Depth**: Include statistics, research findings, and expert quotes."
            ])
        elif intent == 'transactional':
            suggestions.extend([
                "🎯 **Transactional Optimization**: Add clear calls-to-action (CTAs) and benefits.",
                "💡 **Trust**: Include testimonials, guarantees, and security badges."
            ])
        elif intent == 'navigational':
            suggestions.extend([
                "🎯 **Navigational Optimization**: Ensure clear contact information and site structure.",
                "💡 **Accessibility**: Add sitemap, search function, and clear navigation."
            ])
        
        return suggestions

    def calculate_score(self, text, suggestions):
        """Calculate comprehensive content quality score"""
        base_score = 100
        
        # Calculate individual component scores
        length_score = self._calculate_length_score(text)
        keyword_score = self._calculate_keyword_score(text)
        structure_score = self._calculate_structure_score(text) * 100
        readability_score = self._calculate_readability(text) * 100
        
        # Weighted average
        final_score = (
            length_score * 0.25 +
            keyword_score * 0.25 + 
            structure_score * 0.25 +
            readability_score * 0.25
        )
        
        # Deduct for critical issues
        critical_issues = len([s for s in suggestions if any(word in s.lower() for word in ['expand', 'improve', 'add', 'enhance'])])
        final_score -= critical_issues * 3
        
        return max(0, min(100, int(final_score)))

    def _calculate_length_score(self, text):
        word_count = len(text.split())
        if word_count >= 800:
            return 100
        elif word_count >= 500:
            return 80
        elif word_count >= 300:
            return 60
        else:
            return 40

    def _calculate_keyword_score(self, text):
        keywords = self.extract_keywords(text)
        if not keywords:
            return 50
        
        text_lower = text.lower()
        coverage = sum(1 for kw in keywords if kw in text_lower) / len(keywords)
        return coverage * 100

def main():
    st.title("🔍 SmartSEO Hybrid Analyzer")
    st.markdown("Advanced SEO analysis with ML-powered insights")
    
    # Initialize analyzer
    analyzer = HybridSEOAnalyzer()
    
    # Display ML status
    if ML_AVAILABLE:
        st.sidebar.success("✅ ML Features Enabled")
    else:
        st.sidebar.warning("⚠️ Using Rule-Based Analysis")
    
    # Rest of the main function remains the same as app_working.py
    # ... [include the same main() function from app_working.py]

# Include all the same display functions from app_working.py
# ... [include show_overview, show_keyword_analysis, show_intent_ranking, show_suggestions, etc.]

if __name__ == "__main__":
    main()