"""
Corrected Waste Classification System
Based on accurate recycling guidelines
"""

# CORRECTED WASTE CATEGORIES
CORRECTED_WASTE_CATEGORIES = {
    # RECYCLABLE ITEMS
    'recyclable': [
        # Plastics (recyclable types)
        'plastic_water_bottles',      # PET #1 - Recyclable
        'plastic_soda_bottles',       # PET #1 - Recyclable  
        'plastic_detergent_bottles',  # HDPE #2 - Recyclable
        'plastic_food_containers',    # Sometimes recyclable (PET/PP)
        
        # Paper & Cardboard
        'newspaper',                  # Recyclable
        'office_paper',               # Recyclable
        'magazines',                  # Recyclable
        'cardboard_boxes',            # Recyclable
        'cardboard_packaging',        # Recyclable
        
        # Glass
        'glass_beverage_bottles',     # Recyclable
        'glass_food_jars',            # Recyclable
        'glass_cosmetic_containers',  # Recyclable (if clean)
        
        # Metals
        'aluminum_soda_cans',         # Recyclable
        'aluminum_food_cans',         # Recyclable
        'steel_food_cans',            # Recyclable
        'aerosol_cans',               # Recyclable (if empty)
    ],
    
    # NON-RECYCLABLE ITEMS
    'non_recyclable': [
        # Non-recyclable plastics
        'plastic_shopping_bags',      # NOT recyclable curbside
        'plastic_trash_bags',         # NOT recyclable
        'disposable_plastic_cutlery', # NOT recyclable
        'plastic_straws',             # NOT recyclable
        'plastic_cup_lids',           # Sometimes recyclable (PP #5)
        
        # Organic/Compostable
        'food_waste',                 # Compostable, not recyclable
        'eggshells',                  # Compostable
        'coffee_grounds',             # Compostable
        'tea_bags',                   # Compostable
        
        # Textiles
        'clothing',                   # Not recyclable curbside
        'shoes',                      # Not recyclable curbside
        
        # Other non-recyclable
        'paper_cups',                 # Not recyclable (plastic coating)
        'styrofoam_cups',             # Not recyclable
        'styrofoam_food_containers',  # Not recyclable
    ]
}

# DETAILED CLASSIFICATION WITH REASONS
DETAILED_CLASSIFICATION = {
    # RECYCLABLE ITEMS
    'plastic_water_bottles': {
        'category': 'recyclable',
        'type': 'Plastic (PET #1)',
        'reason': 'PET bottles are widely accepted for recycling',
        'disposal': 'Rinse and place in recycling bin'
    },
    'plastic_soda_bottles': {
        'category': 'recyclable', 
        'type': 'Plastic (PET #1)',
        'reason': 'PET bottles are widely accepted for recycling',
        'disposal': 'Rinse and place in recycling bin'
    },
    'plastic_detergent_bottles': {
        'category': 'recyclable',
        'type': 'Plastic (HDPE #2)', 
        'reason': 'HDPE bottles are commonly recycled',
        'disposal': 'Rinse and place in recycling bin'
    },
    'plastic_food_containers': {
        'category': 'recyclable',
        'type': 'Plastic (PET/PP)',
        'reason': 'Often recyclable, check local facilities',
        'disposal': 'Check local recycling rules, rinse if accepted'
    },
    'newspaper': {
        'category': 'recyclable',
        'type': 'Paper',
        'reason': 'Clean, dry paper is highly recyclable',
        'disposal': 'Keep dry, place in paper recycling'
    },
    'office_paper': {
        'category': 'recyclable',
        'type': 'Paper',
        'reason': 'Clean, dry paper is highly recyclable', 
        'disposal': 'Keep dry, place in paper recycling'
    },
    'magazines': {
        'category': 'recyclable',
        'type': 'Paper',
        'reason': 'Magazines are recyclable if clean',
        'disposal': 'Remove plastic wrap, place in paper recycling'
    },
    'cardboard_boxes': {
        'category': 'recyclable',
        'type': 'Cardboard',
        'reason': 'Cardboard is highly recyclable',
        'disposal': 'Flatten, keep dry, place in cardboard recycling'
    },
    'cardboard_packaging': {
        'category': 'recyclable',
        'type': 'Cardboard', 
        'reason': 'Cardboard packaging is recyclable',
        'disposal': 'Remove tape, flatten, keep dry'
    },
    'glass_beverage_bottles': {
        'category': 'recyclable',
        'type': 'Glass',
        'reason': 'Glass bottles are highly recyclable',
        'disposal': 'Rinse, remove caps, place in glass recycling'
    },
    'glass_food_jars': {
        'category': 'recyclable',
        'type': 'Glass',
        'reason': 'Glass jars are recyclable',
        'disposal': 'Rinse, remove lids, place in glass recycling'
    },
    'glass_cosmetic_containers': {
        'category': 'recyclable',
        'type': 'Glass',
        'reason': 'Glass containers are recyclable if clean',
        'disposal': 'Rinse thoroughly, remove pumps/lids'
    },
    'aluminum_soda_cans': {
        'category': 'recyclable',
        'type': 'Metal (Aluminum)',
        'reason': 'Aluminum cans are highly recyclable',
        'disposal': 'Rinse, place in metal recycling'
    },
    'aluminum_food_cans': {
        'category': 'recyclable',
        'type': 'Metal (Aluminum)',
        'reason': 'Aluminum cans are highly recyclable',
        'disposal': 'Rinse, place in metal recycling'
    },
    'steel_food_cans': {
        'category': 'recyclable',
        'type': 'Metal (Steel)',
        'reason': 'Steel cans are recyclable',
        'disposal': 'Rinse, place in metal recycling'
    },
    'aerosol_cans': {
        'category': 'recyclable',
        'type': 'Metal (Steel/Aluminum)',
        'reason': 'Recyclable if completely empty and depressurized',
        'disposal': 'Ensure completely empty, check local rules'
    },
    
    # NON-RECYCLABLE ITEMS
    'plastic_shopping_bags': {
        'category': 'non_recyclable',
        'type': 'Plastic (Film)',
        'reason': 'Not recyclable in curbside programs',
        'disposal': 'Take to special collection points or reuse'
    },
    'plastic_trash_bags': {
        'category': 'non_recyclable',
        'type': 'Plastic (Film)',
        'reason': 'Generally not recyclable',
        'disposal': 'Dispose in general waste'
    },
    'disposable_plastic_cutlery': {
        'category': 'non_recyclable',
        'type': 'Plastic (Small items)',
        'reason': 'Too small and low-grade for recycling',
        'disposal': 'Dispose in general waste'
    },
    'plastic_straws': {
        'category': 'non_recyclable',
        'type': 'Plastic (Small items)',
        'reason': 'Too small for recycling equipment',
        'disposal': 'Dispose in general waste'
    },
    'plastic_cup_lids': {
        'category': 'non_recyclable',
        'type': 'Plastic (PP #5)',
        'reason': 'Often not accepted in curbside recycling',
        'disposal': 'Check local rules, often general waste'
    },
    'food_waste': {
        'category': 'non_recyclable',
        'type': 'Organic/Compostable',
        'reason': 'Should be composted, not recycled',
        'disposal': 'Compost or dispose in general waste'
    },
    'eggshells': {
        'category': 'non_recyclable',
        'type': 'Organic/Compostable',
        'reason': 'Should be composted, not recycled',
        'disposal': 'Compost or dispose in general waste'
    },
    'coffee_grounds': {
        'category': 'non_recyclable',
        'type': 'Organic/Compostable',
        'reason': 'Should be composted, not recycled',
        'disposal': 'Compost or dispose in general waste'
    },
    'tea_bags': {
        'category': 'non_recyclable',
        'type': 'Organic/Compostable',
        'reason': 'Should be composted, not recycled',
        'disposal': 'Compost (remove plastic mesh if present)'
    },
    'clothing': {
        'category': 'non_recyclable',
        'type': 'Textile',
        'reason': 'Not recyclable in curbside programs',
        'disposal': 'Donate, sell, or take to textile recycling'
    },
    'shoes': {
        'category': 'non_recyclable',
        'type': 'Textile/Leather',
        'reason': 'Not recyclable in curbside programs',
        'disposal': 'Donate, sell, or take to specialized recycling'
    },
    'paper_cups': {
        'category': 'non_recyclable',
        'type': 'Paper with plastic coating',
        'reason': 'Plastic coating makes them non-recyclable',
        'disposal': 'Dispose in general waste'
    },
    'styrofoam_cups': {
        'category': 'non_recyclable',
        'type': 'Polystyrene',
        'reason': 'Not accepted in most curbside programs',
        'disposal': 'Dispose in general waste'
    },
    'styrofoam_food_containers': {
        'category': 'non_recyclable',
        'type': 'Polystyrene',
        'reason': 'Not accepted in most curbside programs',
        'disposal': 'Dispose in general waste'
    }
}

def get_corrected_classification(category_name):
    """Get the corrected classification for a waste category"""
    return DETAILED_CLASSIFICATION.get(category_name, {
        'category': 'unknown',
        'type': 'Unknown',
        'reason': 'Category not found in database',
        'disposal': 'Check local recycling guidelines'
    })

def get_recycling_guidance(category_name):
    """Get detailed recycling guidance for a category"""
    info = get_corrected_classification(category_name)
    
    if info['category'] == 'recyclable':
        return f"♻️ **RECYCLABLE** - {info['type']}\n\n**Why:** {info['reason']}\n\n**How to dispose:** {info['disposal']}"
    elif info['category'] == 'non_recyclable':
        return f"🚫 **NON-RECYCLABLE** - {info['type']}\n\n**Why:** {info['reason']}\n\n**How to dispose:** {info['disposal']}"
    else:
        return f"❓ **UNKNOWN** - {info['type']}\n\n**Why:** {info['reason']}\n\n**How to dispose:** {info['disposal']}"

if __name__ == "__main__":
    # Test the corrected classifications
    print("Testing corrected waste classifications:")
    print("=" * 50)
    
    test_categories = [
        'plastic_water_bottles',
        'plastic_shopping_bags', 
        'food_waste',
        'aluminum_soda_cans',
        'styrofoam_cups'
    ]
    
    for category in test_categories:
        print(f"\n{category}:")
        print(get_recycling_guidance(category))
        print("-" * 30)

