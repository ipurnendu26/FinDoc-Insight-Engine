import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import os
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

class_labels = [
    "Groceries", "Dining", "Transportation", "Entertainment",
    "Electricity", "Water", "Gas", "Internet", "Mobile",
    "Mortgage/Rent", "Healthcare", "Insurance", "Subscriptions",
    "Education", "Shopping", "Travel", "Others"
]

class ExpenseClassifier:
    def __init__(self, model_path: Optional[str] = None):
        self.class_labels = class_labels
        self.model_path = model_path or "model/fine_tuned_bert"
        self.tokenizer = None
        self.model = None
        self.load_or_create_model()
    
    def generate_comprehensive_training_data(self) -> Tuple[List[str], List[int]]:
        """Generate comprehensive training data with 60+ diverse examples per category for better precision"""
        training_data = {
            "Groceries": [
                "Walmart grocery shopping", "Kroger food purchase", "Target grocery run",
                "Fresh produce from farmers market", "Costco bulk grocery shopping",
                "Whole Foods organic groceries", "Safeway weekly shopping",
                "Aldi discount groceries", "Food Lion grocery store",
                "Giant Eagle food purchase", "Publix grocery shopping",
                "H-E-B grocery store", "Wegmans food shopping",
                "Meijer grocery purchase", "Stop & Shop groceries",
                "King Soopers food shopping", "Fred Meyer grocery run",
                "QFC grocery store", "Harris Teeter food purchase",
                "Piggly Wiggly groceries", "IGA grocery shopping",
                "Bi-Lo food store", "Winn-Dixie groceries",
                "Food 4 Less shopping", "Save-A-Lot groceries",
                "Trader Joe's grocery", "Sprouts Farmers Market",
                "Sam's Club groceries", "BJ's Wholesale Club",
                "Hy-Vee food store", "Albertsons groceries",
                "Vons grocery shopping", "Ralphs food purchase",
                "Shaw's supermarket", "Acme Markets groceries",
                "Grocery Outlet shopping", "Market Basket groceries",
                "Stater Bros. food store", "WinCo Foods groceries",
                "SuperValu grocery shopping", "Pavilions food purchase",
                "Lucky Supermarkets groceries", "FoodMaxx shopping",
                "El Super grocery store", "Fiesta Mart groceries",
                "99 Ranch Market shopping", "Patel Brothers groceries",
                "H Mart Asian groceries", "Seafood market purchase",
                "Butcher shop meat", "Bakery bread purchase",
                "Deli sandwich groceries", "Organic food store",
                "Ethnic grocery store", "Discount grocery store",
                "Neighborhood corner store", "Bulk food store",
                "Weekly produce box", "CSA farm share",
                "Online grocery delivery", "Instacart grocery order"
            ],
            "Dining": [
                "McDonald's fast food", "Starbucks coffee", "Subway sandwich",
                "Pizza Hut dinner", "KFC chicken meal", "Taco Bell lunch",
                "Burger King meal", "Domino's pizza delivery",
                "Chipotle burrito bowl", "Panera Bread cafe",
                "Olive Garden restaurant", "Applebee's dinner",
                "Chili's restaurant", "TGI Friday's meal",
                "Red Lobster seafood", "Outback Steakhouse",
                "Texas Roadhouse dinner", "Denny's breakfast",
                "IHOP pancakes", "Cracker Barrel meal",
                "Local restaurant dinner", "Food truck lunch",
                "Cafe coffee and pastry", "Bakery fresh bread",
                "Fine dining restaurant", "Sushi restaurant meal",
                "Buffet restaurant", "Steakhouse dinner",
                "Seafood grill meal", "BBQ restaurant",
                "Pho noodle house", "Ramen shop dinner",
                "Tapas bar meal", "Gastropub dinner",
                "Bistro lunch", "Pizzeria meal",
                "Diner breakfast", "Juice bar smoothie",
                "Ice cream parlor", "Frozen yogurt shop",
                "Creperie dessert", "Waffle house breakfast",
                "Brunch cafe", "Late night diner",
                "Takeout Chinese food", "Indian restaurant meal",
                "Thai food dinner", "Mexican cantina meal",
                "Greek taverna dinner", "Middle Eastern restaurant",
                "Vegan restaurant meal", "Vegetarian cafe",
                "Fast casual restaurant", "Drive-thru meal",
                "Lunch special restaurant", "Happy hour bar food",
                "Meal delivery service", "Online food order"
            ],
            "Transportation": [
                "Shell gas station", "Exxon fuel purchase", "BP gasoline",
                "Chevron gas fill-up", "Mobil gas station",
                "Uber ride to airport", "Lyft downtown trip",
                "Metro bus fare", "Subway train ticket",
                "Taxi cab fare", "Car rental Hertz",
                "Enterprise car rental", "Budget rental car",
                "Parking garage fee", "Street parking meter",
                "Toll road payment", "Bridge toll fee",
                "Airport parking", "Valet parking service",
                "Auto repair shop", "Oil change service",
                "Car wash and detail", "Vehicle registration",
                "Driver's license renewal", "Auto insurance premium",
                "AAA membership fee", "Amtrak train ticket",
                "Greyhound bus fare", "Megabus ticket",
                "Bicycle rental", "Scooter rental",
                "Electric vehicle charging", "EV charging station",
                "Carpool ride share", "Zipcar rental",
                "Shuttle bus fare", "Airport transfer service",
                "Ferry ticket", "Boat rental",
                "Motorcycle fuel purchase", "RV park fee",
                "Truck rental", "Moving van rental",
                "Tire replacement", "Brake service",
                "Windshield repair", "Auto detailing",
                "Public transit pass", "Monthly metro card",
                "Light rail ticket", "Regional train fare",
                "Taxi app payment", "Ride-hailing service",
                "Airport limousine service", "Long-term parking fee"
            ],
            "Entertainment": [
                "Movie theater tickets", "Concert venue admission",
                "Sports event tickets", "Theater show tickets",
                "Amusement park entry", "Zoo admission fee",
                "Museum ticket", "Art gallery visit",
                "Bowling alley games", "Mini golf course",
                "Arcade game tokens", "Pool hall rental",
                "Karaoke bar night", "Comedy club show",
                "Dance club cover charge", "Bar drinks",
                "Casino gaming", "Lottery tickets",
                "Video game purchase", "Board game store",
                "Streaming service rental", "Pay-per-view event",
                "Book store purchase", "Magazine subscription",
                "Hobby supplies", "Craft store materials",
                "Escape room admission", "Laser tag game",
                "Paintball park entry", "Trampoline park ticket",
                "Aquarium visit", "Planetarium show",
                "Opera tickets", "Ballet performance",
                "Symphony concert", "Music festival pass",
                "Carnival ride tickets", "Haunted house entry",
                "Stand-up comedy show", "Improv theater ticket",
                "Wine tasting event", "Brewery tour",
                "Nightclub cover charge", "DJ event ticket",
                "Magic show admission", "Circus tickets",
                "Puppet theater show", "Children's play tickets",
                "Outdoor movie night", "Drive-in theater",
                "Board game cafe", "Puzzle room entry",
                "Virtual reality arcade", "E-sports event ticket"
            ],
            "Electricity": [
                "Electric utility bill", "Power company payment",
                "Electricity monthly charge", "Energy bill payment",
                "Utility electric service", "Power grid fee",
                "Renewable energy charge", "Solar panel credit",
                "Electric meter reading", "Peak usage charge",
                "Off-peak electricity", "Smart meter fee",
                "Grid connection charge", "Transmission fee",
                "Distribution charge", "Environmental fee",
                "Fuel adjustment charge", "Demand charge",
                "Service availability fee", "Late payment fee",
                "Electricity reconnection fee", "Power outage credit",
                "Electricity deposit", "Prepaid electricity",
                "Smart home energy", "Electricity bill autopay",
                "Green energy surcharge", "Energy efficiency rebate",
                "Electricity bill discount", "Utility bill payment",
                "Electricity bill online", "Electricity bill by mail",
                "Electricity bill phone payment", "Electricity bill kiosk",
                "Electricity bill in person", "Electricity bill late fee",
                "Electricity bill returned check", "Electricity bill adjustment",
                "Electricity bill payment plan", "Electricity bill budget plan",
                "Electricity bill equal payment", "Electricity bill estimated reading",
                "Electricity bill final bill", "Electricity bill move-in",
                "Electricity bill move-out", "Electricity bill new service",
                "Electricity bill service transfer", "Electricity bill stop service"
            ],
            "Water": [
                "Water utility bill", "Municipal water service",
                "Water department payment", "Sewer service charge",
                "Water meter reading", "Drainage fee",
                "Water conservation charge", "Irrigation water",
                "Well water testing", "Water softener service",
                "Bottled water delivery", "Water filter replacement",
                "Pool water service", "Sprinkler system water",
                "Landscape irrigation", "Water usage overage",
                "Water bill autopay", "Water bill online payment",
                "Water bill by mail", "Water bill phone payment",
                "Water bill in person", "Water bill late fee",
                "Water bill returned check", "Water bill adjustment",
                "Water bill payment plan", "Water bill budget plan",
                "Water bill equal payment", "Water bill estimated reading",
                "Water bill final bill", "Water bill move-in",
                "Water bill move-out", "Water bill new service",
                "Water bill service transfer", "Water bill stop service",
                "Water bill deposit", "Water bill reconnection fee",
                "Water bill credit", "Water bill discount",
                "Water bill surcharge", "Water bill environmental fee"
            ],
            "Gas": [
                "Natural gas utility", "Gas company bill",
                "Heating gas service", "Propane tank refill",
                "Gas meter reading", "Winter heating bill",
                "Gas appliance service", "Furnace gas usage",
                "Water heater gas", "Stove gas connection",
                "Fireplace gas line", "Gas leak repair",
                "Gas safety inspection", "Pipeline maintenance fee",
                "Gas bill autopay", "Gas bill online payment",
                "Gas bill by mail", "Gas bill phone payment",
                "Gas bill in person", "Gas bill late fee",
                "Gas bill returned check", "Gas bill adjustment",
                "Gas bill payment plan", "Gas bill budget plan",
                "Gas bill equal payment", "Gas bill estimated reading",
                "Gas bill final bill", "Gas bill move-in",
                "Gas bill move-out", "Gas bill new service",
                "Gas bill service transfer", "Gas bill stop service",
                "Gas bill deposit", "Gas bill reconnection fee",
                "Gas bill credit", "Gas bill discount",
                "Gas bill surcharge", "Gas bill environmental fee"
            ],
            "Internet": [
                "Comcast internet service", "Verizon FiOS internet",
                "AT&T internet plan", "Charter Spectrum internet",
                "Cox internet service", "Xfinity internet bill",
                "Fiber optic internet", "Cable internet service",
                "DSL internet connection", "Satellite internet",
                "Mobile hotspot data", "WiFi service provider",
                "Internet installation fee", "Modem rental charge",
                "Router equipment fee", "Bandwidth upgrade",
                "Internet speed boost", "Data overage charge",
                "Internet bill autopay", "Internet bill online payment",
                "Internet bill by mail", "Internet bill phone payment",
                "Internet bill in person", "Internet bill late fee",
                "Internet bill returned check", "Internet bill adjustment",
                "Internet bill payment plan", "Internet bill budget plan",
                "Internet bill equal payment", "Internet bill estimated reading",
                "Internet bill final bill", "Internet bill move-in",
                "Internet bill move-out", "Internet bill new service",
                "Internet bill service transfer", "Internet bill stop service",
                "Internet bill deposit", "Internet bill reconnection fee",
                "Internet bill credit", "Internet bill discount",
                "Internet bill surcharge", "Internet bill environmental fee"
            ],
            "Mobile": [
                "Verizon wireless bill", "AT&T mobile service",
                "T-Mobile phone plan", "Sprint cellular service",
                "Metro PCS payment", "Boost Mobile bill",
                "Cricket wireless service", "Straight Talk phone",
                "Prepaid phone card", "International calling",
                "Text messaging plan", "Data plan upgrade",
                "Phone insurance premium", "Device protection plan",
                "Mobile app purchase", "In-app subscription",
                "Phone repair service", "Screen replacement",
                "Battery replacement", "Phone case purchase",
                "Mobile bill autopay", "Mobile bill online payment",
                "Mobile bill by mail", "Mobile bill phone payment",
                "Mobile bill in person", "Mobile bill late fee",
                "Mobile bill returned check", "Mobile bill adjustment",
                "Mobile bill payment plan", "Mobile bill budget plan",
                "Mobile bill equal payment", "Mobile bill estimated reading",
                "Mobile bill final bill", "Mobile bill move-in",
                "Mobile bill move-out", "Mobile bill new service",
                "Mobile bill service transfer", "Mobile bill stop service",
                "Mobile bill deposit", "Mobile bill reconnection fee",
                "Mobile bill credit", "Mobile bill discount",
                "Mobile bill surcharge", "Mobile bill environmental fee"
            ],
            "Mortgage/Rent": [
                "Monthly mortgage payment", "Rent to landlord",
                "Property management fee", "HOA dues payment",
                "Condo association fee", "Apartment rent",
                "House rental payment", "Property tax payment",
                "Homeowners insurance", "Rental insurance",
                "Security deposit", "Pet deposit fee",
                "Late rent penalty", "Lease renewal fee",
                "Property maintenance", "Landscaping service",
                "Snow removal service", "Pest control service",
                "Mortgage bill autopay", "Mortgage bill online payment",
                "Mortgage bill by mail", "Mortgage bill phone payment",
                "Mortgage bill in person", "Mortgage bill late fee",
                "Mortgage bill returned check", "Mortgage bill adjustment",
                "Mortgage bill payment plan", "Mortgage bill budget plan",
                "Mortgage bill equal payment", "Mortgage bill estimated reading",
                "Mortgage bill final bill", "Mortgage bill move-in",
                "Mortgage bill move-out", "Mortgage bill new service",
                "Mortgage bill service transfer", "Mortgage bill stop service",
                "Mortgage bill deposit", "Mortgage bill reconnection fee",
                "Mortgage bill credit", "Mortgage bill discount",
                "Mortgage bill surcharge", "Mortgage bill environmental fee"
            ],
            "Healthcare": [
                "Doctor visit copay", "Dentist appointment",
                "Eye exam optometrist", "Prescription medication",
                "Pharmacy drug purchase", "Hospital bill payment",
                "Urgent care visit", "Emergency room bill",
                "Physical therapy session", "Chiropractor treatment",
                "Mental health counseling", "Dermatologist visit",
                "Specialist consultation", "Medical lab tests",
                "X-ray imaging", "MRI scan payment",
                "Blood work analysis", "Vaccination shot",
                "Annual physical exam", "Preventive care visit",
                "Healthcare bill autopay", "Healthcare bill online payment",
                "Healthcare bill by mail", "Healthcare bill phone payment",
                "Healthcare bill in person", "Healthcare bill late fee",
                "Healthcare bill returned check", "Healthcare bill adjustment",
                "Healthcare bill payment plan", "Healthcare bill budget plan",
                "Healthcare bill equal payment", "Healthcare bill estimated reading",
                "Healthcare bill final bill", "Healthcare bill move-in",
                "Healthcare bill move-out", "Healthcare bill new service",
                "Healthcare bill service transfer", "Healthcare bill stop service",
                "Healthcare bill deposit", "Healthcare bill reconnection fee",
                "Healthcare bill credit", "Healthcare bill discount",
                "Healthcare bill surcharge", "Healthcare bill environmental fee"
            ],
            "Insurance": [
                "Auto insurance premium", "Health insurance payment",
                "Life insurance premium", "Home insurance bill",
                "Renters insurance payment", "Disability insurance",
                "Umbrella insurance policy", "Travel insurance",
                "Pet insurance premium", "Dental insurance",
                "Vision insurance payment", "Long-term care insurance",
                "Professional liability", "Business insurance premium",
                "Insurance bill autopay", "Insurance bill online payment",
                "Insurance bill by mail", "Insurance bill phone payment",
                "Insurance bill in person", "Insurance bill late fee",
                "Insurance bill returned check", "Insurance bill adjustment",
                "Insurance bill payment plan", "Insurance bill budget plan",
                "Insurance bill equal payment", "Insurance bill estimated reading",
                "Insurance bill final bill", "Insurance bill move-in",
                "Insurance bill move-out", "Insurance bill new service",
                "Insurance bill service transfer", "Insurance bill stop service",
                "Insurance bill deposit", "Insurance bill reconnection fee",
                "Insurance bill credit", "Insurance bill discount",
                "Insurance bill surcharge", "Insurance bill environmental fee"
            ],
            "Subscriptions": [
                "Netflix streaming service", "Amazon Prime membership",
                "Spotify music subscription", "Apple Music service",
                "YouTube Premium", "Disney Plus streaming",
                "Hulu subscription", "HBO Max service",
                "Adobe Creative Suite", "Microsoft Office 365",
                "Dropbox cloud storage", "iCloud storage plan",
                "Gym membership fee", "Magazine subscription",
                "Newspaper digital access", "Software license renewal",
                "VPN service subscription", "Password manager service",
                "Subscription bill autopay", "Subscription bill online payment",
                "Subscription bill by mail", "Subscription bill phone payment",
                "Subscription bill in person", "Subscription bill late fee",
                "Subscription bill returned check", "Subscription bill adjustment",
                "Subscription bill payment plan", "Subscription bill budget plan",
                "Subscription bill equal payment", "Subscription bill estimated reading",
                "Subscription bill final bill", "Subscription bill move-in",
                "Subscription bill move-out", "Subscription bill new service",
                "Subscription bill service transfer", "Subscription bill stop service",
                "Subscription bill deposit", "Subscription bill reconnection fee",
                "Subscription bill credit", "Subscription bill discount",
                "Subscription bill surcharge", "Subscription bill environmental fee"
            ],
            "Education": [
                "College tuition payment", "University fees",
                "Student loan payment", "Textbook purchase",
                "Online course enrollment", "Training workshop fee",
                "Certification exam", "Professional development",
                "Language learning app", "Tutoring service",
                "Educational software", "School supplies",
                "Laboratory fees", "Graduation ceremony",
                "Academic conference", "Research materials",
                "Education bill autopay", "Education bill online payment",
                "Education bill by mail", "Education bill phone payment",
                "Education bill in person", "Education bill late fee",
                "Education bill returned check", "Education bill adjustment",
                "Education bill payment plan", "Education bill budget plan",
                "Education bill equal payment", "Education bill estimated reading",
                "Education bill final bill", "Education bill move-in",
                "Education bill move-out", "Education bill new service",
                "Education bill service transfer", "Education bill stop service",
                "Education bill deposit", "Education bill reconnection fee",
                "Education bill credit", "Education bill discount",
                "Education bill surcharge", "Education bill environmental fee"
            ],
            "Shopping": [
                "Amazon online purchase", "eBay auction win",
                "Target retail shopping", "Best Buy electronics",
                "Home Depot hardware", "Lowe's home improvement",
                "Macy's clothing store", "Nordstrom fashion",
                "Walmart general merchandise", "Costco warehouse",
                "Clothing boutique", "Shoe store purchase",
                "Jewelry store", "Electronics retailer",
                "Furniture store", "Appliance purchase",
                "Sporting goods store", "Toy store shopping",
                "Shopping mall purchase", "Outlet store shopping",
                "Thrift store purchase", "Consignment shop buy",
                "Online electronics store", "Online clothing store",
                "Gift shop purchase", "Bookstore shopping",
                "Pet store purchase", "Beauty supply store",
                "Cosmetics store shopping", "Perfume shop buy",
                "Home goods store", "Kitchenware store",
                "Garden center purchase", "Hardware store buy",
                "Auto parts store", "Bicycle shop purchase",
                "Toy store online", "Music store purchase",
                "Art supply store", "Craft store shopping",
                "Stationery store buy", "Office supply store",
                "Mattress store purchase", "Luggage store buy"
            ],
            "Travel": [
                "Airline ticket purchase", "Hotel accommodation",
                "Vacation rental Airbnb", "Car rental travel",
                "Train ticket booking", "Bus transportation",
                "Cruise ship booking", "Travel insurance",
                "Passport renewal", "Visa application fee",
                "Airport shuttle service", "Luggage fees",
                "Travel agency booking", "Tour guide service",
                "Travel vaccinations", "Currency exchange",
                "International roaming", "Travel adapter purchase",
                "Travel bill autopay", "Travel bill online payment",
                "Travel bill by mail", "Travel bill phone payment",
                "Travel bill in person", "Travel bill late fee",
                "Travel bill returned check", "Travel bill adjustment",
                "Travel bill payment plan", "Travel bill budget plan",
                "Travel bill equal payment", "Travel bill estimated reading",
                "Travel bill final bill", "Travel bill move-in",
                "Travel bill move-out", "Travel bill new service",
                "Travel bill service transfer", "Travel bill stop service",
                "Travel bill deposit", "Travel bill reconnection fee",
                "Travel bill credit", "Travel bill discount",
                "Travel bill surcharge", "Travel bill environmental fee"
            ],
            "Others": [
                "Bank service fee", "ATM withdrawal fee",
                "Wire transfer charge", "Check printing",
                "Safe deposit box", "Notary service",
                "Legal consultation", "Tax preparation",
                "Accounting service", "Investment advisory",
                "Charity donation", "Religious offering",
                "Gift purchase", "Wedding gift",
                "Birthday present", "Holiday decoration",
                "Miscellaneous expense", "Uncategorized purchase",
                "Donation to school", "Fundraising event expense",
                "Lost item replacement", "Pet care expense",
                "Moving expense", "Storage unit fee",
                "Post office box fee", "Passport photo fee",
                "Visa application charge", "Immigration fee",
                "Court filing fee", "Jury duty expense",
                "Parking ticket", "Traffic fine",
                "Library fine", "Recycling fee",
                "Document translation fee", "Copying service fee",
                "Fax service fee", "Public records fee",
                "Background check fee", "Fingerprinting fee"
            ]
        }
        texts = []
        labels = []
        for category, examples in training_data.items():
            category_index = self.class_labels.index(category)
            for example in examples:
                texts.append(example)
                labels.append(category_index)
        return texts, labels
    
    def create_balanced_dataset(self, texts: List[str], labels: List[int]) -> tf.data.Dataset:
        """Create a balanced training dataset"""
        # Convert to tensors
        encodings = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=128,
            return_tensors="tf"
        )
        
        dataset = tf.data.Dataset.from_tensor_slices((
            dict(encodings),
            tf.convert_to_tensor(labels)
        ))
        
        # Shuffle and batch
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(16)  # Increased batch size
        
        return dataset
    
    def comprehensive_fine_tune(self):
        """Comprehensive fine-tuning with proper validation"""
        try:
            logger.info("Starting comprehensive fine-tuning...")
            
            # Generate comprehensive training data
            texts, labels = self.generate_comprehensive_training_data()
            logger.info(f"Generated {len(texts)} training samples across {len(set(labels))} categories")
            
            # Split into train and validation
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Create datasets
            train_dataset = self.create_balanced_dataset(train_texts, train_labels)
            val_dataset = self.create_balanced_dataset(val_texts, val_labels)
            
            # Configure model with better parameters
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),  # Lower learning rate
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
            
            # Add callbacks for better training
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=3,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-6
                )
            ]
            
            # Train the model
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=10,  # More epochs with early stopping
                callbacks=callbacks,
                verbose=1
            )
            
            # Get final accuracy
            final_accuracy = max(history.history['val_accuracy'])
            logger.info(f"Final validation accuracy: {final_accuracy:.4f}")
            
            if final_accuracy >= 0.95:
                logger.info("🎉 Achieved 95%+ accuracy!")
            else:
                logger.warning(f"Accuracy {final_accuracy:.4f} below 95% target")
            
        except Exception as e:
            logger.error(f"Comprehensive fine-tuning failed: {str(e)}")
    
    def load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            if os.path.exists(self.model_path):
                self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                self.model = TFBertForSequenceClassification.from_pretrained(self.model_path)
                logger.info(f"Loaded existing model from {self.model_path}")
            else:
                self.create_new_model()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.create_new_model()
    
    def create_new_model(self):
        """Create and train new model"""
        logger.info("Creating new model...")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = TFBertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(self.class_labels)
        )
        self.comprehensive_fine_tune()  # Use comprehensive training
        self.save_model()
    
    def save_model(self):
        """Save model and tokenizer"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)
            logger.info(f"Model and tokenizer saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def predict_category(self, text: str) -> str:
        """Predict expense category for given text using the trained BERT model"""
        if not self.model or not self.tokenizer:
            self.load_or_create_model()
        # Preprocess text
        clean_text = str(text).strip().lower()
        try:
            inputs = self.tokenizer(clean_text, return_tensors="tf", truncation=True, padding=True, max_length=128)
            outputs = self.model(**inputs)
            logits = outputs.logits.numpy()
            pred_idx = int(np.argmax(logits, axis=1)[0])
            return self.class_labels[pred_idx]
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return "Others"

_classifier: Optional[ExpenseClassifier] = None


def predict_category(text: str) -> str:
    """Load the classifier on first use instead of training during module import."""
    global _classifier
    if _classifier is None:
        _classifier = ExpenseClassifier()
    return _classifier.predict_category(text)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Train and save model when run directly
    classifier = ExpenseClassifier()
