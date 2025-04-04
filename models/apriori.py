import pandas as pd
import numpy as np
from collections import defaultdict
import logging
import time
from itertools import combinations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)
class AprioriAnalysis:
    def __init__(self, min_support=0.01, min_confidence=0.5, max_length=None):
        self.min_support=min_support
        self.min_confidence=min_confidence
        self.max_length=max_length
        self.frequent_itemsets=None
        self.association_rules=None
        self.item_mapping=None
        self.reverse_mapping=None
        self.products_info=None

    def _create_item_mapping(self, baskets):
        unique_items=set()
        for basket in baskets:
            unique_items.update(basket)
        item_mapping={item: idx for idx, item in enumerate(unique_items)}
        reverse_mapping={idx: item for item, idx in item_mapping.items()}
        return item_mapping, reverse_mapping
    def _transform_baskets(self, baskets, item_mapping):
        return [set(item_mapping[item] for item in basket) for basket in baskets]

    def _find_frequent_itemsets(self, baskets, min_support_count):
        item_counts=defaultdict(int)
        for basket in baskets:
            for item in basket:
                item_counts[frozenset([item])] += 1
        L1={itemset: count for itemset, count in item_counts.items() if count >= min_support_count}
        frequent_itemsets={1: L1}
        current_L=L1
        k=2
        max_len=self.max_length if self.max_length is not None else float('inf')
        while current_L and k <= max_len:
            start_time=time.time()
            Ck=self._generate_candidates(current_L, k)
            candidate_counts=defaultdict(int)
            for basket in baskets:
                for candidate in self._find_candidates_in_basket(basket, Ck, k):
                    candidate_counts[candidate] += 1
            current_L={itemset: count for itemset, count in candidate_counts.items() if count >= min_support_count}
            if current_L:
                frequent_itemsets[k]=current_L
            logger.info(f"Found {len(current_L)} frequent {k}-itemsets in {time.time() - start_time:.2f} seconds")
            k += 1

        return frequent_itemsets

    def _generate_candidates(self, Lk_minus_1, k):
        candidates=set()
        prev_itemsets=list(Lk_minus_1.keys())
        for i in range(len(prev_itemsets)):
            for j in range(i + 1, len(prev_itemsets)):
                set1=set(prev_itemsets[i])
                set2=set(prev_itemsets[j])
                if len(set1.union(set2)) == k:
                    candidate=frozenset(set1.union(set2))
                    all_subsets_frequent=True
                    for subset in [frozenset(s) for s in combinations(candidate, k - 1)]:
                        if subset not in Lk_minus_1:
                            all_subsets_frequent=False
                            break
                    if all_subsets_frequent:
                        candidates.add(candidate)

        return candidates

    def _find_candidates_in_basket(self, basket, candidates, k):
        if len(basket) < k:
            return []
        return [candidate for candidate in candidates if candidate.issubset(basket)]

    def _generate_association_rules(self, frequent_itemsets, total_transactions):
        rules=[]
        flat_itemsets={}
        for k, k_itemsets in frequent_itemsets.items():
            if k > 1:
                for itemset, count in k_itemsets.items():
                    flat_itemsets[itemset]=count
        for itemset, itemset_count in flat_itemsets.items():
            itemset_support=itemset_count / total_transactions
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent=frozenset(antecedent)
                    consequent=frozenset(itemset - antecedent)
                    antecedent_count=0
                    for k, k_itemsets in frequent_itemsets.items():
                        if k == len(antecedent):
                            antecedent_count=k_itemsets.get(antecedent, 0)
                            break
                    confidence=itemset_count / antecedent_count
                    if confidence < self.min_confidence:
                        continue
                    consequent_count=0
                    for k, k_itemsets in frequent_itemsets.items():
                        if k == len(consequent):
                            consequent_count=k_itemsets.get(consequent, 0)
                            break
                    consequent_support=consequent_count / total_transactions
                    lift=confidence / consequent_support
                    rules.append({
                        'antecedent': antecedent,
                        'consequent': consequent,
                        'support': itemset_support,
                        'confidence': confidence,
                        'lift': lift,
                        'antecedent_support': antecedent_count / total_transactions,
                        'consequent_support': consequent_support
                    })

        return rules

    def _map_items_back(self, frequent_itemsets, association_rules):
        mapped_itemsets={}
        for k, k_itemsets in frequent_itemsets.items():
            mapped_itemsets[k]={}
            for itemset, count in k_itemsets.items():
                mapped_itemset=frozenset(self.reverse_mapping[item] for item in itemset)
                mapped_itemsets[k][mapped_itemset]=count
        mapped_rules=[]
        for rule in association_rules:
            mapped_rule=rule.copy()
            mapped_rule['antecedent']=frozenset(self.reverse_mapping[item] for item in rule['antecedent'])
            mapped_rule['consequent']=frozenset(self.reverse_mapping[item] for item in rule['consequent'])
            mapped_rules.append(mapped_rule)

        return mapped_itemsets, mapped_rules

    def fit(self, baskets, products_info=None):
        start_time=time.time()
        logger.info(
            f"Starting Apriori analysis with min_support={self.min_support}, min_confidence={self.min_confidence}")
        self.products_info=products_info
        self.item_mapping, self.reverse_mapping=self._create_item_mapping(baskets)
        logger.info(f"Created mapping for {len(self.item_mapping)} unique items")
        transformed_baskets=self._transform_baskets(baskets, self.item_mapping)
        logger.info(f"Transformed {len(baskets)} baskets")
        total_transactions=len(baskets)
        min_support_count=int(self.min_support * total_transactions)
        logger.info(f"Minimum support count: {min_support_count} out of {total_transactions} transactions")
        frequent_itemsets=self._find_frequent_itemsets(transformed_baskets, min_support_count)
        logger.info(f"Found frequent itemsets for {len(frequent_itemsets)} different sizes")
        association_rules=self._generate_association_rules(frequent_itemsets, total_transactions)
        logger.info(f"Generated {len(association_rules)} association rules")
        self.frequent_itemsets, self.association_rules=self._map_items_back(frequent_itemsets, association_rules)
        logger.info(f"Apriori analysis completed in {time.time() - start_time:.2f} seconds")
        return self
    def get_frequent_itemsets(self, with_product_names=False):
        if self.frequent_itemsets is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        result=[]
        for k, k_itemsets in self.frequent_itemsets.items():
            for itemset, count in k_itemsets.items():
                item_info={
                    'itemset': itemset,
                    'length': k,
                    'support_count': count
                }
                if with_product_names and self.products_info:
                    item_info['items']=[
                        self.products_info.get(product_id, {'name': f'Product {product_id}'})
                        for product_id in itemset
                    ]

                result.append(item_info)
        result.sort(key=lambda x: (-x['support_count'], x['length']))

        return result

    def get_association_rules(self, with_product_names=False, sort_by='lift'):
        if self.association_rules is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        result=[]
        for rule in self.association_rules:
            rule_info=rule.copy()
            if with_product_names and self.products_info:
                rule_info['antecedent_items']=[
                    self.products_info.get(product_id, {'name': f'Product {product_id}'})
                    for product_id in rule['antecedent']
                ]
                rule_info['consequent_items']=[
                    self.products_info.get(product_id, {'name': f'Product {product_id}'})
                    for product_id in rule['consequent']
                ]
            result.append(rule_info)
        if sort_by == 'support':
            result.sort(key=lambda x: -x['support'])
        elif sort_by == 'confidence':
            result.sort(key=lambda x: -x['confidence'])
        else:
            result.sort(key=lambda x: -x['lift'])
        return result

    def get_product_recommendations(self, basket, top_n=5):
        if self.association_rules is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        basket=set(basket)
        applicable_rules=[]
        for rule in self.association_rules:
            if rule['antecedent'].issubset(basket):
                new_items=rule['consequent'] - basket
                if new_items:
                    for item in new_items:
                        applicable_rules.append({
                            'product_id': item,
                            'score': rule['lift'],
                            'confidence': rule['confidence'],
                            'from_rule': rule
                        })
        applicable_rules.sort(key=lambda x: -x['score'])
        seen_products=set()
        recommendations=[]
        for rule in applicable_rules:
            product_id=rule['product_id']
            if product_id not in seen_products:
                seen_products.add(product_id)
                if self.products_info:
                    rule['product_info']=self.products_info.get(product_id, {'name': f'Product {product_id}'})

                recommendations.append(rule)
                if len(recommendations) >= top_n:
                    break

        return recommendations

    def find_bundle_opportunities(self, min_lift=1.5, min_confidence=0.5, top_n=10):
        if self.association_rules is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        bundle_candidates=[
            rule for rule in self.association_rules
            if rule['lift'] >= min_lift and rule['confidence'] >= min_confidence
        ]
        bundle_candidates.sort(key=lambda x: -x['lift'])
        bundle_opportunities=[]
        for i, rule in enumerate(bundle_candidates[:top_n]):
            bundle={
                'id': i + 1,
                'products': set(rule['antecedent']).union(set(rule['consequent'])),
                'lift': rule['lift'],
                'confidence': rule['confidence'],
                'support': rule['support'],
                'rule': rule
            }
            if self.products_info:
                bundle['product_info']=[
                    self.products_info.get(product_id, {'name': f'Product {product_id}'})
                    for product_id in bundle['products']
                ]

            bundle_opportunities.append(bundle)

        return bundle_opportunities
def test_apriori_analysis():
    import random
    products={
        1: {'name': 'Laptop', 'category': 'Electronics'},
        2: {'name': 'Mouse', 'category': 'Electronics'},
        3: {'name': 'Keyboard', 'category': 'Electronics'},
        4: {'name': 'Monitor', 'category': 'Electronics'},
        5: {'name': 'Headphones', 'category': 'Electronics'},
        6: {'name': 'Smartphone', 'category': 'Electronics'},
        7: {'name': 'Tablet', 'category': 'Electronics'},
        8: {'name': 'Desk', 'category': 'Furniture'},
        9: {'name': 'Chair', 'category': 'Furniture'},
        10: {'name': 'Lamp', 'category': 'Furniture'}
    }
    baskets=[]
    n_transactions=1000
    for i in range(n_transactions):
        basket=set()
        r=random.random()

        if r < 0.3:
            basket.update([1, 2, 3])
            if random.random() < 0.4:
                basket.add(4)
        elif r < 0.55:
            basket.update([5, 6])
            if random.random() < 0.3:
                basket.add(7)
        elif r < 0.75:
            basket.update([8, 9, 10])
        else:
            n_items=random.randint(1, 5)
            basket.update(random.sample(range(1, 11), n_items))
        baskets.append(basket)
    apriori=AprioriAnalysis(min_support=0.1, min_confidence=0.5)
    apriori.fit(baskets, products)
    frequent_itemsets=apriori.get_frequent_itemsets(with_product_names=True)
    print(f"Found {len(frequent_itemsets)} frequent itemsets")
    print("\nTop 5 frequent itemsets:")
    for i, itemset in enumerate(frequent_itemsets[:5]):
        item_names=[product['name'] for product in itemset.get('items', [])]
        print(f"{i + 1}. {', '.join(item_names)} (support: {itemset['support_count'] / n_transactions:.2f})")
    rules=apriori.get_association_rules(with_product_names=True)
    print(f"\nFound {len(rules)} association rules")
    print("\nTop 5 association rules by lift:")
    for i, rule in enumerate(rules[:5]):
        antecedent_names=[product['name'] for product in rule.get('antecedent_items', [])]
        consequent_names=[product['name'] for product in rule.get('consequent_items', [])]
        print(f"{i + 1}. {' & '.join(antecedent_names)} → {' & '.join(consequent_names)}")
        print(f"   Support: {rule['support']:.2f}, Confidence: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f}")
    test_basket={1, 2}
    recommendations=apriori.get_product_recommendations(test_basket, top_n=3)
    print("\nRecommendations for a basket with Laptop and Mouse:")
    for i, rec in enumerate(recommendations):
        product_name=rec.get('product_info', {}).get('name', f"Product {rec['product_id']}")
        print(f"{i + 1}. {product_name} (score: {rec['score']:.2f}, confidence: {rec['confidence']:.2f})")
    bundles=apriori.find_bundle_opportunities(min_lift=1.5, min_confidence=0.5, top_n=3)
    print("\nTop 3 bundle opportunities:")
    for i, bundle in enumerate(bundles):
        product_names=[product['name'] for product in bundle.get('product_info', [])]
        print(f"{i + 1}. {', '.join(product_names)}")
        print(f"   Lift: {bundle['lift']:.2f}, Confidence: {bundle['confidence']:.2f}")
    print("\nApriori analysis testing completed.")
if __name__ == "__main__":
    test_apriori_analysis()