""""
PROJECT DESCRIPTION

Build and maintain the database of a SaaS company.
The project uses python, SQLite and fastAPI

"""

from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel
import json
import sqlite3
import requests
app = FastAPI()

# -------------------------------MODELS---------------------------------------------------------------
# https://fastapi.tiangolo.com/tutorial/response-model/


class Company(BaseModel):
    CO_NAME: str
    CO_ADDRESS: str
    CO_VAT: int
    CO_BANKACCOUNT: str


class Customer(BaseModel):
    CUS_NAME: str
    CUS_EMAIL: str
    CUS_ADDRESS: str
    CUS_PHONE: str
    COMPANY_ID: int


class Subscription(BaseModel):
    SUB_PRICE: float
    SUB_CURRENCY: str
    COMPANY_ID: int


class Quote(BaseModel):
    QUOTE_QUANTITY: int
    QUOTE_ACTIVE: bool = False
    QUOTE_TOTALPRICEEUR: float
    CUSTOMER_ID: int
    SUBSCRIPTION_ID: int


class AcceptQuote(BaseModel):
    ACCEPT: bool = True
    QUOTE_ID: int


class Invoice(BaseModel):
    INV_PENDING: bool = True
    INV_TOTALPRICEEUR: float
    QUOTE_ID: int


class PayInvoice(BaseModel):
    CREDIT_CARD: int
    QUOTE_ID: int

# -------------------------------CREATE TABLES---------------------------------------------------------------


def createTables():
    dbase = sqlite3.connect("valentin_mets_nous_20.db",
                            isolation_level=None, check_same_thread=False)

    dbase.execute(''' CREATE TABLE IF NOT EXISTS company(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        co_name TEXT NOT NULL,
        co_address TEXT NOT NULL,
        co_VAT INT NOT NULL,
        co_bankaccount TEXT NOT NULL) ''')

    dbase.execute(''' CREATE TABLE IF NOT EXISTS customer(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cus_name TEXT NOT NULL,
        cus_email TEXT NOT NULL,
        cus_address TEXT NOT NULL,
        cus_phone TEXT NOT NULL,
        company_id INTEGER NOT NULL,
        FOREIGN KEY (company_id) REFERENCES company(id)
        ) ''')

    dbase.execute(''' CREATE TABLE IF NOT EXISTS subscription(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sub_price FLOAT NOT NULL,
        sub_currency TEXT NOT NULL,
        company_id INTEGER NOT NULL,
        FOREIGN KEY (company_id) REFERENCES company(id)
        ) ''')

    dbase.execute(''' CREATE TABLE IF NOT EXISTS quote(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        quote_quantity INT NOT NULL,
        quote_active BOOL NOT NULL,
        total_price_eur FLOAT,
        customer_id INTEGER NOT NULL,
        subscription_id INTEGER NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customer(id),
        FOREIGN KEY (subscription_id) REFERENCES subscription(id)
        ) ''')

    dbase.execute(''' CREATE TABLE IF NOT EXISTS invoice(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        inv_pending BOOL NOT NULL,
        total_price_eur FLOAT,
        quote_id INTEGER NOT NULL,
        FOREIGN KEY (quote_id) REFERENCES quote(id)
        ) ''')

    dbase.close()

# -------------------------------FUNCTIONS---------------------------------------------------------------


def calc_vat(price):
    vat = price / 100 * 21
    vatPrice = price + vat
    roundedVATP = round(vatPrice, 2)
    return(roundedVATP)


def calc_conversion(price, currency):
    if (currency == "EUR"):
        return(price)
    response = requests.get(
        "https://v6.exchangerate-api.com/v6/e1ab50496b040f58ab530bc6/latest/EUR")
    currentCurrencies = response.json()
    conversionRate = currentCurrencies["conversion_rates"][currency]
    convPrice = (1 / conversionRate) * price
    roundedConvP = round(convPrice, 2)
    return(roundedConvP)


def check_credit_card(cardNumbers):
    cardNumbers = str(cardNumbers)
    if (len(cardNumbers) != 16):
        return(False)
    checkingDigit = int(cardNumbers[:1])
    cardNumbers = cardNumbers[:-1]
    cardNumbers = cardNumbers[::-1]
    # https://www.geeksforgeeks.org/python-convert-number-to-list-of-integers/
    # https://stackoverflow.com/questions/522563/accessing-the-index-in-for-loops?fbclid=IwAR1cGYMdlDWiIBYptJD5YmRPL7XW_RR203NefQGqHBgkRugzlfVHxsyaFDY
    cardNumbers = list(map(int, cardNumbers))
    for index, num in enumerate(cardNumbers):
        if (index % 2 == 0):
            num = num * 2
            if (num > 9):
                num = num - 9
                cardNumbers[index] = num
    finalNumber = sum(cardNumbers) + checkingDigit
    if (finalNumber % 10 == 0):
        return(True)
    return (False)

# -------------------------------PLATFORM---------------------------------------------------------------


@app.get("/")
def root():
    createTables()
    return {"message": "Welcome Valentin to your favourite SaaS platform ! It's nice to have you back."}


@app.post("/create-company-account")
def insert_company(company: Company):
    dbase = sqlite3.connect("valentin_mets_nous_20.db",
                            isolation_level=None, check_same_thread=False)
    new_company = dict(company)
    dbase.execute('''
        INSERT INTO company(co_name, co_address, co_VAT, co_bankaccount)
        VALUES(?,?,?,?)''', (new_company["CO_NAME"], new_company["CO_ADDRESS"], new_company["CO_VAT"], new_company["CO_BANKACCOUNT"]))
    dbase.close()
    return {"message": "Company account successfully created."}


@app.post("/create-customer-account")
def insert_customer(customer: Customer):
    dbase = sqlite3.connect("valentin_mets_nous_20.db",
                            isolation_level=None, check_same_thread=False)
    new_customer = dict(customer)
    dbase.execute('''
            INSERT INTO customer (cus_name, cus_email, cus_address, cus_phone, company_id)
            VALUES(?,?,?,?,?)''', (new_customer["CUS_NAME"], new_customer["CUS_EMAIL"], new_customer["CUS_ADDRESS"], new_customer["CUS_PHONE"], new_customer["COMPANY_ID"]))
    dbase.close()
    return {"message": "Customer account successfully created."}


@app.post("/create-subscripton")
def insert_subscription(subscription: Subscription):
    dbase = sqlite3.connect("valentin_mets_nous_20.db",
                            isolation_level=None, check_same_thread=False)
    new_sub = dict(subscription)
    dbase.execute('''
            INSERT INTO subscription (sub_price, sub_currency, company_id)
            VALUES(?,?,?)''', (new_sub["SUB_PRICE"], new_sub["SUB_CURRENCY"], new_sub["COMPANY_ID"]))
    dbase.close()
    return {"message": "Subscription successfully created."}


@app.post("/create-quote")
def insert_quote(quote: Quote):
    dbase = sqlite3.connect("valentin_mets_nous_20.db",
                            isolation_level=None, check_same_thread=False)
    new_quote = dict(quote)
    subprice = dbase.execute('''
            SELECT sub_price FROM subscription WHERE ID = ?''', [new_quote["SUBSCRIPTION_ID"]]).fetchall()[0][0]
    subcurrency = dbase.execute('''
            SELECT sub_currency FROM subscription WHERE ID = ?''', [new_quote["SUBSCRIPTION_ID"]]).fetchall()[0][0]
    totalprice = int(new_quote["QUOTE_QUANTITY"]) * \
        float(subprice)
    if (subcurrency != "EUR"):
        eurprice = calc_conversion(totalprice, subcurrency)
        vateurprice = calc_vat(eurprice)
    elif (subcurrency == "EUR"):
        eurprice = totalprice
        vateurprice = calc_vat(eurprice)

    query_result = dbase.execute('''
            INSERT INTO quote (quote_quantity, quote_active, total_price_eur, customer_id, subscription_id)
            VALUES(?,?,?,?,?)''', (new_quote["QUOTE_QUANTITY"], new_quote["QUOTE_ACTIVE"], eurprice, new_quote["CUSTOMER_ID"], new_quote["SUBSCRIPTION_ID"]))
    # https://www.kite.com/python/docs/sqlite3.Cursor.lastrowid
    quote_id = int(query_result.lastrowid)

    new_quote["QUOTE_ID"] = quote_id
    totalprice = int(new_quote["QUOTE_QUANTITY"]) * \
        float(subprice)
    if (subcurrency != "EUR"):
        new_quote["QUOTE_TOTAL_PRICE_EUR"] = calc_conversion(
            totalprice, subcurrency)
        new_quote["QUOTE_TOTAL_PRICE_VAT_INCL"] = calc_vat(
            new_quote["QUOTE_TOTAL_PRICE_EUR"])
    elif (subcurrency == "EUR"):
        new_quote["QUOTE_TOTAL_PRICE_VAT_INCL"] = calc_vat(totalprice)
    dbase.close()
    return (new_quote)


@app.post("/accept-quote")
def accept_quote(acceptQuote: AcceptQuote):
    dbase = sqlite3.connect("valentin_mets_nous_20.db",
                            isolation_level=None, check_same_thread=False)
    new_acceptquote = dict(acceptQuote)
    if (new_acceptquote["ACCEPT"] == True):
        dbase.execute('''
            UPDATE quote SET quote_active = ? WHERE id = ?''', (new_acceptquote["ACCEPT"], new_acceptquote["QUOTE_ID"]))
        totalpriceeur = dbase.execute('''
            SELECT total_price_eur FROM quote WHERE id = ?''', [new_acceptquote["QUOTE_ID"]]).fetchall()[0][0]
        dbase.execute('''
            INSERT INTO invoice (INV_PENDING,TOTAL_PRICE_EUR, QUOTE_ID) 
            VALUES(?,?,?)''', (True, totalpriceeur, new_acceptquote["QUOTE_ID"]))
        dbase.close()
        return {"message": "Quote " + str(new_acceptquote["QUOTE_ID"]) + " has been accepted, your pending invoice awaits !"}
    else:
        return {"result": "Oops, there has been a mistake. Try again."}


@app.get("/check-payment")
def check_payment(quote_id: int):
    dbase = sqlite3.connect("valentin_mets_nous_20.db",
                            isolation_level=None, check_same_thread=False)
    quoteofinv = dict({"QUOTE_ID": quote_id})
    inv_tobechecked = dbase.execute('''
        SELECT * FROM invoice WHERE QUOTE_ID = ?''', [quoteofinv["QUOTE_ID"]]).fetchall()
    dbase.close()
    if (len(inv_tobechecked) == 1):
        tupleinv = list(inv_tobechecked[0])
        if (tupleinv[1] == 1):
            return {"message": "Invoice is pending."}
        elif (tupleinv[1] == 0):
            return {"message": "Invoice is already paid."}
    else:
        return {"message": "No invoice found with the quote id : " + str(quoteofinv["QUOTE_ID"])}


@app.post("/pay-invoice")
def pay_invoice(payInvoice: PayInvoice):
    dbase = sqlite3.connect("valentin_mets_nous_20.db",
                            isolation_level=None, check_same_thread=False)
    new_payinv = dict(payInvoice)
    if (check_credit_card(new_payinv["CREDIT_CARD"]) == True):
        dbase.execute('''
            UPDATE invoice SET inv_pending = FALSE WHERE quote_id = ?''', [new_payinv["QUOTE_ID"]])
        dbase.close()
        return{"message": "Payment accepted."}
    else:
        return {"message": "Oops, payment refused... Wrong credit card : " + str(new_payinv["CREDIT_CARD"])}


@app.post("/reactivate-invoice")
def reactivate_invoice(invoice_id: int):
    dbase = sqlite3.connect("valentin_mets_nous_20.db",
                            isolation_level=None, check_same_thread=False)
    inv_tobereactivated = dict({"INVOICE_ID": invoice_id})
    dbase.execute('''
        UPDATE invoice SET inv_pending = TRUE WHERE ID = ?''', [inv_tobereactivated["INVOICE_ID"]])
    dbase.close
    return{"message": "The invoice has been successfully reactivated and is pending again."}


@app.post("/delete-invoice")
def delete_invoice(invoice_id: int):
    dbase = sqlite3.connect("valentin_mets_nous_20.db",
                            isolation_level=None, check_same_thread=False)
    inv_tobedeleted = dict({"INVOICE_ID": invoice_id})
    dbase.execute('''
        DELETE FROM invoice WHERE id = ?''', [inv_tobedeleted["INVOICE_ID"]])
    dbase.close
    return{"message": "Invoice " + str(invoice_id) + " has been successfully deleted."}

# -------------------------------ANALYTICS---------------------------------------------------------------


@app.get("/get-monthly-recurring-revenue")
def monthly_recurring_revenue():
    dbase = sqlite3.connect("valentin_mets_nous_20.db",
                            isolation_level=None, check_same_thread=False)
    invoices_all = dbase.execute('''
        SELECT total_price_eur FROM invoice''').fetchall()
    list_of_prices = [item for t in invoices_all for item in t]
    MRR = sum(list_of_prices)
    dbase.close
    return {"message": "Total MRR : " + str(MRR) + " EUR."}


@app.get("/get-annual-recurring-revenue")
def annual_recurring_revenue(subscription_id: int):
    dbase = sqlite3.connect("valentin_mets_nous_20.db",
                            isolation_level=None, check_same_thread=False)
    sub_toARR = dict({"SUBSCRIPTION_ID": subscription_id})
    subprice = dbase.execute('''
        SELECT sub_price FROM subscription WHERE ID = ?''', [sub_toARR["SUBSCRIPTION_ID"]]).fetchall()[0][0]
    subcurrency = dbase.execute('''
        SELECT sub_currency FROM subscription WHERE ID = ?''', [sub_toARR["SUBSCRIPTION_ID"]]).fetchall()[0][0]
    subpriceeur = calc_conversion(subprice, subcurrency)
    ARR = float(subpriceeur)*12
    dbase.close
    return {"message": "Total ARR : " + str(ARR) + " EUR."}


@app.get("/get-total-customer")
def get_total_customer():
    dbase = sqlite3.connect("valentin_mets_nous_20.db",
                            isolation_level=None, check_same_thread=False)
    totalcustomers = dbase.execute(
        '''SELECT COUNT(*) FROM customer''').fetchall()[0][0]
    dbase.close
    return{"message": "Total number of customers : " + str(totalcustomers)}


@app.get("/average-revenue-per-customer")
def get_average_revenue_per_customer(customer_id: int):
    dbase = sqlite3.connect("valentin_mets_nous_20.db",
                            isolation_level=None, check_same_thread=False)
    analysed_customer = dict({"CUSTOMER_ID": customer_id})
    tupleofquotes = dbase.execute('''
        SELECT total_price_eur FROM quote WHERE quote_active = TRUE AND customer_id = ?''', [analysed_customer["CUSTOMER_ID"]]).fetchall()
    list_of_revenues = [item for t in tupleofquotes for item in t]
    total_rev_cus = sum(list_of_revenues)
    nrofquotes = dbase.execute('''
        SELECT COUNT(*) FROM quote WHERE quote_active = TRUE AND customer_id = ?''', [analysed_customer["CUSTOMER_ID"]]).fetchall()[0][0]
    ARC = float(total_rev_cus) / int(nrofquotes)
    roundedARC = round(ARC, 2)
    dbase.close
    return {"message": "Total Average Revenue for customer " + str(customer_id) + " is " + str(roundedARC) + " EUR."}


@app.get("/all-customers")
def get_all_customers():
    dbase = sqlite3.connect("valentin_mets_nous_20.db",
                            isolation_level=None, check_same_thread=False)
    jointtables = dbase.execute('''
    SELECT customer.cus_name AS 'Customer', quote.subscription_id AS 'Subscription ID', quote.total_price_eur AS 'Price in EUR'
        FROM customer INNER JOIN quote ON customer.id=quote.customer_id WHERE quote.quote_active = TRUE''').fetchall()
    return jointtables

# ----------------------------------------------------------------------------------------------


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# -------------------------------THE END---------------------------------------------------------------
