from shared_functions import *

# Global variable to store loaded food items
food_items = []

def main():
    """Main function for interactive CLI food recommendation system"""
    try:
        print("Interactive Food Recommendation System")
        print("=" * 50)
        print("Loading food database...")
        
        # Load food data from file
        global food_items
        food_items = load_food_data('/home/project/FoodDataSet.json')
        print(f"Loaded {len(food_items)} food items successfully")

        collection = create_similarity_search_collection(
            "interactive_food_search",
            {"description" : "A collection for interactive food search."}
        )
        print(f"Collection created successfully.")

        ## Storing the food data docs in collection with their embeddings
        populate_similarity_collection(collection, food_items)
    except Exception as error:
        print(f"Error initializing system: {error}")


if __name__ == '__main__':
    main()
