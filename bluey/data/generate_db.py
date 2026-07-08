import sqlite3
import random
from datetime import timedelta
from faker import Faker
import os

# Configuration
DB_PATH = os.path.join(os.path.dirname(__file__), 'insurance.db')
NUM_RECORDS_BASE = 150  # Minimum 100 rows required
fake = Faker('en_US')  # Data generator in English

def create_connection():
    """Create a connection to the SQLite database."""
    try:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH) # Remove existing DB to start from scratch
        conn = sqlite3.connect(DB_PATH)
        return conn
    except sqlite3.Error as e:
        print(f"Connection error: {e}")
        return None

def create_tables(conn):
    """Create database tables."""
    cursor = conn.cursor()
    
    # 1. Regions
    cursor.execute('''
        CREATE TABLE Regions (
            RegionID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL UNIQUE
        )
    ''')
    
    # 2. Addresses
    cursor.execute('''
        CREATE TABLE Addresses (
            AddressID INTEGER PRIMARY KEY AUTOINCREMENT,
            Street TEXT,
            City TEXT,
            ZipCode TEXT,
            State TEXT,
            Country TEXT DEFAULT 'Italy'
        )
    ''')

    # 3. Offices
    cursor.execute('''
        CREATE TABLE Offices (
            OfficeID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT,
            RegionID INTEGER,
            AddressID INTEGER,
            Phone TEXT,
            FOREIGN KEY (RegionID) REFERENCES Regions(RegionID),
            FOREIGN KEY (AddressID) REFERENCES Addresses(AddressID)
        )
    ''')

    # 4. Agents
    cursor.execute('''
        CREATE TABLE Agents (
            AgentID INTEGER PRIMARY KEY AUTOINCREMENT,
            FirstName TEXT,
            LastName TEXT,
            Email TEXT UNIQUE,
            Phone TEXT,
            OfficeID INTEGER,
            HireDate DATE,
            FOREIGN KEY (OfficeID) REFERENCES Offices(OfficeID)
        )
    ''')

    # 5. Customers
    cursor.execute('''
        CREATE TABLE Customers (
            CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
            FirstName TEXT,
            LastName TEXT,
            Email TEXT,
            Phone TEXT,
            BirthDate DATE,
            FiscalCode TEXT UNIQUE,
            AddressID INTEGER,
            FOREIGN KEY (AddressID) REFERENCES Addresses(AddressID)
        )
    ''')

    # 6. ProductTypes
    cursor.execute('''
        CREATE TABLE ProductTypes (
            ProductTypeID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT UNIQUE
        )
    ''')

    # 7. Products
    cursor.execute('''
        CREATE TABLE Products (
            ProductID INTEGER PRIMARY KEY AUTOINCREMENT,
            ProductTypeID INTEGER,
            Name TEXT,
            Description TEXT,
            BasePremium REAL,
            Active BOOLEAN DEFAULT 1,
            FOREIGN KEY (ProductTypeID) REFERENCES ProductTypes(ProductTypeID)
        )
    ''')

    # 8. Coverages
    cursor.execute('''
        CREATE TABLE Coverages (
            CoverageID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT,
            Description TEXT,
            MinLimit REAL,
            MaxLimit REAL
        )
    ''')

    # 9. ProductCoverages
    cursor.execute('''
        CREATE TABLE ProductCoverages (
            ProductCoverageID INTEGER PRIMARY KEY AUTOINCREMENT,
            ProductID INTEGER,
            CoverageID INTEGER,
            IsMandatory BOOLEAN,
            FOREIGN KEY (ProductID) REFERENCES Products(ProductID),
            FOREIGN KEY (CoverageID) REFERENCES Coverages(CoverageID)
        )
    ''')

    # 10. Discounts
    cursor.execute('''
        CREATE TABLE Discounts (
            DiscountID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT,
            Description TEXT,
            Percentage REAL
        )
    ''')

    # 11. Policies
    cursor.execute('''
        CREATE TABLE Policies (
            PolicyID INTEGER PRIMARY KEY AUTOINCREMENT,
            CustomerID INTEGER,
            AgentID INTEGER,
            ProductID INTEGER,
            PolicyNumber TEXT UNIQUE,
            StartDate DATE,
            EndDate DATE,
            TotalPremium REAL,
            Status TEXT, -- Active, Expired, Cancelled
            FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),
            FOREIGN KEY (AgentID) REFERENCES Agents(AgentID),
            FOREIGN KEY (ProductID) REFERENCES Products(ProductID)
        )
    ''')

    # 12. PolicyCoverages
    cursor.execute('''
        CREATE TABLE PolicyCoverages (
            PolicyCoverageID INTEGER PRIMARY KEY AUTOINCREMENT,
            PolicyID INTEGER,
            CoverageID INTEGER,
            LimitAmount REAL,
            PremiumPart REAL,
            FOREIGN KEY (PolicyID) REFERENCES Policies(PolicyID),
            FOREIGN KEY (CoverageID) REFERENCES Coverages(CoverageID)
        )
    ''')

    # 13. PolicyDiscounts
    cursor.execute('''
        CREATE TABLE PolicyDiscounts (
            PolicyDiscountID INTEGER PRIMARY KEY AUTOINCREMENT,
            PolicyID INTEGER,
            DiscountID INTEGER,
            AppliedAmount REAL,
            FOREIGN KEY (PolicyID) REFERENCES Policies(PolicyID),
            FOREIGN KEY (DiscountID) REFERENCES Discounts(DiscountID)
        )
    ''')

    # 14. Beneficiaries
    cursor.execute('''
        CREATE TABLE Beneficiaries (
            BeneficiaryID INTEGER PRIMARY KEY AUTOINCREMENT,
            PolicyID INTEGER,
            Name TEXT,
            Relation TEXT, -- Spouse, Child, Parent, Other
            AllocationPercentage REAL,
            FOREIGN KEY (PolicyID) REFERENCES Policies(PolicyID)
        )
    ''')

    # 15. AutoRiskDetails
    cursor.execute('''
        CREATE TABLE AutoRiskDetails (
            AutoRiskID INTEGER PRIMARY KEY AUTOINCREMENT,
            PolicyID INTEGER,
            PlateNumber TEXT,
            VIN TEXT,
            Make TEXT,
            Model TEXT,
            Year INTEGER,
            FOREIGN KEY (PolicyID) REFERENCES Policies(PolicyID)
        )
    ''')

    # 16. HomeRiskDetails
    cursor.execute('''
        CREATE TABLE HomeRiskDetails (
            HomeRiskID INTEGER PRIMARY KEY AUTOINCREMENT,
            PolicyID INTEGER,
            AddressID INTEGER,
            SquareMeters INTEGER,
            BuildYear INTEGER,
            PropertyType TEXT, -- Apartment, House, Villa
            FOREIGN KEY (PolicyID) REFERENCES Policies(PolicyID),
            FOREIGN KEY (AddressID) REFERENCES Addresses(AddressID)
        )
    ''')

    # 17. Payments
    cursor.execute('''
        CREATE TABLE Payments (
            PaymentID INTEGER PRIMARY KEY AUTOINCREMENT,
            PolicyID INTEGER,
            Amount REAL,
            PaymentDate DATE,
            Method TEXT, -- CreditCard, BankTransfer, Cash
            Status TEXT, -- Paid, Pending, Failed
            TransactionID TEXT,
            FOREIGN KEY (PolicyID) REFERENCES Policies(PolicyID)
        )
    ''')

    # 18. Claims
    cursor.execute('''
        CREATE TABLE Claims (
            ClaimID INTEGER PRIMARY KEY AUTOINCREMENT,
            PolicyID INTEGER,
            ClaimNumber TEXT UNIQUE,
            IncidentDate DATE,
            FileDate DATE,
            Description TEXT,
            Status TEXT, -- Open, Closed, Rejected, Investigating
            AmountClaimed REAL,
            AmountPaid REAL,
            FOREIGN KEY (PolicyID) REFERENCES Policies(PolicyID)
        )
    ''')

    # 19. ClaimItems
    cursor.execute('''
        CREATE TABLE ClaimItems (
            ClaimItemID INTEGER PRIMARY KEY AUTOINCREMENT,
            ClaimID INTEGER,
            Description TEXT,
            EstimatedCost REAL,
            ApprovedCost REAL,
            FOREIGN KEY (ClaimID) REFERENCES Claims(ClaimID)
        )
    ''')

    # 20. CustomerInteractions
    cursor.execute('''
        CREATE TABLE CustomerInteractions (
            InteractionID INTEGER PRIMARY KEY AUTOINCREMENT,
            CustomerID INTEGER,
            AgentID INTEGER,
            InteractionDate DATE,
            Type TEXT, -- Call, Email, Meeting
            Notes TEXT,
            FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),
            FOREIGN KEY (AgentID) REFERENCES Agents(AgentID)
        )
    ''')
    
    conn.commit()
    print("Tables created successfully.")

def generate_data(conn):
    """Populate the database with synthetic data."""
    cursor = conn.cursor()
    
    # 1. Regions
    regions = ['North Italy', 'Central Italy', 'South Italy', 'Islands']
    for r in regions:
        cursor.execute("INSERT INTO Regions (Name) VALUES (?)", (r,))
    
    # 2. Addresses (many more than customers for risks etc)
    address_ids = []
    print("Generating Addresses...")
    for _ in range(NUM_RECORDS_BASE * 3):
        cursor.execute('''
            INSERT INTO Addresses (Street, City, ZipCode, State, Country)
            VALUES (?, ?, ?, ?, ?)
        ''', (fake.street_address(), fake.city(), fake.postcode(), fake.current_country_code(), 'Italy'))
        address_ids.append(cursor.lastrowid)

    # 3. Offices
    office_ids = []
    print("Generating Offices...")
    for _ in range(20): # 20 offices
        region_id = random.randint(1, len(regions))
        addr_id = random.choice(address_ids)
        cursor.execute("INSERT INTO Offices (Name, RegionID, AddressID, Phone) VALUES (?, ?, ?, ?)",
                       (f"Office {fake.city()}", region_id, addr_id, fake.phone_number()))
        office_ids.append(cursor.lastrowid)

    # 4. Agents
    agent_ids = []
    print("Generating Agents...")
    for _ in range(50):
        cursor.execute('''
            INSERT INTO Agents (FirstName, LastName, Email, Phone, OfficeID, HireDate)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (fake.first_name(), fake.last_name(), fake.unique.email(), fake.phone_number(), 
              random.choice(office_ids), fake.date_between(start_date='-10y', end_date='today')))
        agent_ids.append(cursor.lastrowid)

    # 5. Customers
    customer_ids = []
    print("Generating Customers...")
    for _ in range(NUM_RECORDS_BASE):
        addr_id = random.choice(address_ids)
        cursor.execute('''
            INSERT INTO Customers (FirstName, LastName, Email, Phone, BirthDate, FiscalCode, AddressID)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (fake.first_name(), fake.last_name(), fake.unique.email(), fake.phone_number(), 
              fake.date_of_birth(minimum_age=18, maximum_age=90), fake.ssn(), addr_id))
        customer_ids.append(cursor.lastrowid)

    # 6. ProductTypes
    types = ['Auto', 'Home', 'Life', 'Health', 'Travel']
    type_ids = []
    for t in types:
        cursor.execute("INSERT INTO ProductTypes (Name) VALUES (?)", (t,))
        type_ids.append(cursor.lastrowid)

    # 7. Products
    products_data = [
        ('Auto Liability Basic', 1, 'Mandatory base coverage', 300.0),
        ('Auto Comprehensive', 1, 'Full vehicle coverage', 800.0),
        ('Safe Home', 2, 'Fire and theft protection', 150.0),
        ('Whole Life', 3, 'Life policy in case of death', 200.0),
        ('Health Plus', 4, 'Medical expense reimbursement', 500.0),
        ('Travel Relax', 5, 'Assistance and baggage', 50.0)
    ]
    product_ids = []
    for p in products_data:
        cursor.execute("INSERT INTO Products (Name, ProductTypeID, Description, BasePremium) VALUES (?, ?, ?, ?)", p)
        product_ids.append(cursor.lastrowid)

    # 8. Coverages
    coverages = [
        ('Civil Liability', 5000000, 10000000),
        ('Fire and Theft', 5000, 500000),
        ('Glass Damage', 500, 2000),
        ('Vandalism', 1000, 10000),
        ('Driver Injury', 10000, 100000),
        ('Roadside Assistance', 100, 500),
        ('Family Liability', 100000, 1000000),
        ('Medical Expenses', 1000, 50000),
        ('Permanent Disability', 50000, 500000),
        ('Death', 50000, 1000000)
    ]
    coverage_ids = []
    for c in coverages:
        cursor.execute("INSERT INTO Coverages (Name, MinLimit, MaxLimit) VALUES (?, ?, ?)", c)
        coverage_ids.append(cursor.lastrowid)

    # 9. ProductCoverages (reasonable random mapping)
    for pid in product_ids:
        num_covs = random.randint(2, 5)
        selected_covs = random.sample(coverage_ids, num_covs)
        for cid in selected_covs:
            cursor.execute('''
                INSERT INTO ProductCoverages (ProductID, CoverageID, IsMandatory) 
                VALUES (?, ?, ?)
            ''', (pid, cid, random.choice([0, 1])))

    # 10. Discounts
    discounts = [
        ('Web Discount', 0.10),
        ('Loyalty Discount', 0.15),
        ('Multi-product Discount', 0.05),
        ('Young Driver Discount', 0.05),
        ('No Discount', 0.0)
    ]
    discount_ids = []
    for d in discounts:
        cursor.execute("INSERT INTO Discounts (Name, Percentage) VALUES (?, ?)", d)
        discount_ids.append(cursor.lastrowid)

    # 11. Policies
    policy_ids = []
    print("Generating Policies...")
    for _ in range(NUM_RECORDS_BASE * 2): # average 2 policies per customer
        start_date = fake.date_between(start_date='-2y', end_date='today')
        end_date = start_date + timedelta(days=365)
        price_mul = random.uniform(0.8, 1.5)
        
        prod_id = random.choice(product_ids)
        cursor.execute("SELECT BasePremium FROM Products WHERE ProductID = ?", (prod_id,))
        base_prem = cursor.fetchone()[0]
        
        cursor.execute('''
            INSERT INTO Policies (CustomerID, AgentID, ProductID, PolicyNumber, StartDate, EndDate, TotalPremium, Status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (random.choice(customer_ids), random.choice(agent_ids), prod_id, 
              fake.unique.bothify(text='POL-####-????'), start_date, end_date, 
              round(base_prem * price_mul, 2), random.choice(['Active', 'Active', 'Expired'])))
        policy_ids.append(cursor.lastrowid)

    # 12. PolicyCoverages & 13. PolicyDiscounts
    for pid in policy_ids:
        # Coverages
        cursor.execute("SELECT ProductID FROM Policies WHERE PolicyID = ?", (pid,))
        pol_prod_id = cursor.fetchone()[0]
        cursor.execute("SELECT CoverageID FROM ProductCoverages WHERE ProductID = ?", (pol_prod_id,))
        avail_cov_ids = [row[0] for row in cursor.fetchall()]
        
        for cid in avail_cov_ids:
            cursor.execute("INSERT INTO PolicyCoverages (PolicyID, CoverageID, LimitAmount, PremiumPart) VALUES (?, ?, ?, ?)",
                           (pid, cid, random.randint(5000, 50000), random.uniform(20, 100)))
        
        # Discounts
        if random.random() > 0.5:
            did = random.choice(discount_ids)
            cursor.execute("INSERT INTO PolicyDiscounts (PolicyID, DiscountID, AppliedAmount) VALUES (?, ?, ?)",
                           (pid, did, random.uniform(10, 50)))

    # 14. Beneficiaries (only for life policies - approx product type 3)
    cursor.execute("SELECT PolicyID FROM Policies WHERE ProductID IN (SELECT ProductID FROM Products WHERE ProductTypeID = 3)")
    life_policy_ids = [row[0] for row in cursor.fetchall()]
    for lpid in life_policy_ids:
        cursor.execute("INSERT INTO Beneficiaries (PolicyID, Name, Relation, AllocationPercentage) VALUES (?, ?, ?, ?)",
                       (lpid, fake.name(), random.choice(['Spouse', 'Child', 'Parent']), 100.0))

    # 15. AutoRiskDetails (only for auto policies - approx product type 1)
    cursor.execute("SELECT PolicyID FROM Policies WHERE ProductID IN (SELECT ProductID FROM Products WHERE ProductTypeID = 1)")
    auto_policy_ids = [row[0] for row in cursor.fetchall()]
    for apid in auto_policy_ids:
        cursor.execute('''
            INSERT INTO AutoRiskDetails (PolicyID, PlateNumber, VIN, Make, Model, Year)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (apid, fake.license_plate(), fake.vin(), fake.word(), fake.word(), random.randint(2000, 2024)))

    # 16. HomeRiskDetails (only for home policies - approx product type 2)
    cursor.execute("SELECT PolicyID FROM Policies WHERE ProductID IN (SELECT ProductID FROM Products WHERE ProductTypeID = 2)")
    home_policy_ids = [row[0] for row in cursor.fetchall()]
    for hpid in home_policy_ids:
        cursor.execute('''
            INSERT INTO HomeRiskDetails (PolicyID, AddressID, SquareMeters, BuildYear, PropertyType)
            VALUES (?, ?, ?, ?, ?)
        ''', (hpid, random.choice(address_ids), random.randint(50, 300), random.randint(1950, 2023), 
              random.choice(['Apartment', 'House', 'Villa'])))

    # 17. Payments
    print("Generating Payments...")
    for pid in policy_ids:
        for _ in range(random.randint(1, 4)): # 1-4 payments per policy
            cursor.execute("SELECT TotalPremium FROM Policies WHERE PolicyID = ?", (pid,))
            prem = cursor.fetchone()[0]
            cursor.execute('''
                INSERT INTO Payments (PolicyID, Amount, PaymentDate, Method, Status, TransactionID)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (pid, round(prem / 2, 2), fake.date_between(start_date='-1y', end_date='today'),
                  random.choice(['CreditCard', 'BankTransfer']), 'Paid', fake.uuid4()))

    # 18. Claims & 19. ClaimItems
    print("Generating Claims...")
    for _ in range(int(len(policy_ids) * 0.2)): # 20% of policies have claims
        pid = random.choice(policy_ids)
        cursor.execute('''
            INSERT INTO Claims (PolicyID, ClaimNumber, IncidentDate, FileDate, Description, Status, AmountClaimed, AmountPaid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pid, fake.unique.bothify(text='CLM-####-????'), fake.date_between(start_date='-1y', end_date='today'),
              fake.date_between(start_date='-1y', end_date='today'), fake.sentence(), 
              random.choice(['Open', 'Closed', 'Paid']), random.uniform(500, 50000), random.uniform(0, 50000)))
        claim_id = cursor.lastrowid
        
        # Items
        for _ in range(random.randint(1, 3)):
            cursor.execute('''
                INSERT INTO ClaimItems (ClaimID, Description, EstimatedCost, ApprovedCost)
                VALUES (?, ?, ?, ?)
            ''', (claim_id, fake.word(), random.uniform(100, 5000), random.uniform(0, 5000)))

    # 20. CustomerInteractions
    print("Generating Interactions...")
    for _ in range(NUM_RECORDS_BASE * 5):
        cursor.execute('''
            INSERT INTO CustomerInteractions (CustomerID, AgentID, InteractionDate, Type, Notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (random.choice(customer_ids), random.choice(agent_ids), 
              fake.date_between(start_date='-2y', end_date='today'),
              random.choice(['Call', 'Email', 'Meeting']), fake.sentence()))

    conn.commit()
    print(f"Database populated successfully: {len(policy_ids)} policies generated.")

if __name__ == '__main__':
    conn = create_connection()
    if conn:
        create_tables(conn)
        generate_data(conn)
        conn.close()
