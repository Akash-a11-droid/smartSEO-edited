from ml.keyword_extractor import HybridKeywordExtractor
from ml.preprocessing import preprocess
import nltk
import re
from collections import Counter

class ContentScorer:
    def __init__(self):
        self.weights = {
            'length': 0.2,
            'readability': 0.15,
            'keyword_coverage': 0.25,
            'keyword_frequency': 0.15,
            'intent_alignment': 0.15,
            'ranking_potential': 0.1
        }
    
    def calculate_score(self, suggestions):
        base_score = 100
        
        # Deduction rules for different types of suggestions
        deduction_rules = {
            "Content is short": -15,
            "Average sentence length is high": -10,
            "Add missing important keywords": -5,  # per missing keyword
            "Keyword.*appears only once": -3,      # per keyword
            "Predicted ranking score is low": -20
        }
        
        for suggestion in suggestions:
            for pattern, deduction in deduction_rules.items():
                if re.search(pattern, suggestion):
                    base_score += deduction
                    break
        
        return max(0, min(100, base_score))  # Ensure score is between 0-100

class ContentImprover:
    def __init__(self, keyword_extractor=None):
        self.keyword_extractor = keyword_extractor or HybridKeywordExtractor()
        self.scorer = ContentScorer()

    def suggest_improvements(self, text, predicted_intent, predicted_rank, target_keywords=None):
        suggestions = []

        # Preprocess
        processed = preprocess(text)
        tokens = nltk.word_tokenize(processed)
        word_count = len(tokens)
        sentences = nltk.sent_tokenize(text)

        # 1. Content length
        if word_count < 300:
            suggestions.append(
                f"Content is short ({word_count} words). Top-ranking articles often exceed 800 words — consider expanding with examples, FAQs, or case studies."
            )
        elif word_count > 2000:
            suggestions.append(
                f"Content is very long ({word_count} words). Consider breaking it into multiple articles or adding a table of contents for better user experience."
            )

        # 2. Readability
        if sentences:
            avg_sentence_length = word_count / len(sentences)
            if avg_sentence_length > 20:
                suggestions.append("Average sentence length is high — break down long sentences for better readability.")
            if avg_sentence_length < 8:
                suggestions.append("Average sentence length is very short — consider combining some sentences for better flow.")

        # 3. Keyword coverage
        extracted = self.keyword_extractor.extract_keywords(text, top_k=5)
        if target_keywords:
            missing = [kw for kw in target_keywords if kw not in extracted]
            if missing:
                suggestions.append(f"Add missing important keywords: {', '.join(missing)}")

        # 4. Keyword frequency
        text_lower = text.lower()
        for kw in extracted[:5]:  # Check top 5 keywords only
            count = text_lower.count(kw.lower())
            if count < 2:
                suggestions.append(f"Keyword '{kw}' appears only once — consider using it naturally a few more times.")
            elif count > 10:
                suggestions.append(f"Keyword '{kw}' appears {count} times — may be over-optimized, consider reducing frequency.")

        # 5. Intent alignment
        if predicted_intent == "informational":
            suggestions.append("Since intent is informational, add headings (H2/H3), step-by-step guides, or FAQs.")
        elif predicted_intent == "transactional":
            suggestions.append("Since intent is transactional, add strong CTAs (Buy Now, Contact Us) and product details.")
        elif predicted_intent == "navigational":
            suggestions.append("Since intent is navigational, ensure clear internal links and site structure.")

        # 6. Ranking feedback
        if predicted_rank and len(predicted_rank) > 0:
            rank_score = predicted_rank[0]
            if rank_score > 10:
                suggestions.append("Predicted ranking score is low — improve by adding backlinks, meta descriptions, and keyword density.")
            elif rank_score <= 5:
                suggestions.append("Predicted ranking score is good — focus on maintaining quality and adding fresh content.")

        # 7. Structure analysis
        if len(sentences) > 0:
            # Check if text has headings
            has_headings = any(re.search(r'<h[1-6]|^#+ |^[A-Z][A-Z\s]{10,}:', line) for line in text.split('\n'))
            if not has_headings and len(sentences) > 5:
                suggestions.append("Content lacks headings — add structure with H2/H3 tags to improve readability and SEO.")

        return suggestions

    def get_content_score(self, text, suggestions):
        return self.scorer.calculate_score(suggestions)