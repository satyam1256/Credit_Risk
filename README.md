# Credit-Risk-Modelling-Using-Machine-Learning

## **Objective of the project**

**1. Understanding Credit Risk Modelling:**
Gain a comprehensive understanding of the fundamentals of credit risk assessment and discover the pivotal role that machine learning plays in modelling credit risk.

**2. Data Preprocessing and Feature Engineering:**
Delve into various data preprocessing techniques and feature engineering methods essential for preparing data suitable for credit risk modelling.

**3. Building Machine Learning Models for Credit Risk:**
Learn to implement various machine learning algorithms like XGBoost, DecisionTrees and Random Forest to develop precise and predictive models for assessing credit risk.

**4. Evaluation and Interpretation of Credit Risk Models:**
Master the skills needed to evaluate credit risk models and interpret their outcomes, enabling informed decision-making within a financial context.

# **DataSet used :**

There are 2 dataset used. The case_study_1 is the Bureau dataset and case_study_2 is the Internal product dataset.

**Bureau Dataset:**

**Source:** The Bureau dataset typically refers to credit information obtained from credit bureaus such as Equifax, Experian, TransUnion, etc. These bureaus collect data from various financial institutions, creditors, and other sources to create credit reports for individuals and businesses.

**Data:** This dataset includes information like credit scores, payment history, outstanding debt, credit inquiries, public records (such as bankruptcies or liens), and other relevant credit-related information.

**Usage:** Lenders use this data to assess an individual's or business's creditworthiness, determine the risk of default on a loan, set interest rates, and make decisions on approving or denying credit applications.

**Internal Product Dataset:**

**Source:** The Internal product dataset, on the other hand, refers to data that is specific to the financial institution or organization using it. This could include data related to the institution's own products, services, customer interactions, and operational processes.

**Data:** This dataset may include information on internal credit scores, customer account details, transaction history, customer behavior patterns, product usage information, and any other data collected by the institution in the course of its operations.

**Usage:** Financial institutions use internal product data to complement the information obtained from external credit bureaus. By combining internal and external data, institutions can gain a more comprehensive view of a customer's financial behavior and creditworthiness, allowing for more accurate risk assessments and tailored product offerings.

In summary, while the Bureau dataset provides external credit information from credit bureaus, the Internal product dataset offers internal data specific to the institution using it. Combining these datasets can provide a more holistic view of credit risk and help financial institutions make well-informed lending decisions.

# Credit Risk Modeling Using Machine Learning

## Banking Termninoligies

### Asset

- **Asset is something that can give some profit to bank finally. (Bank favour)**
- Known as Loan Product

#### Examples

- Housing loan
- Personal loan
- Vehicle loan
- Group loan
- Education loan
- Credit Card

### Liability

- **Liability is something that can give some Loss to bank finally. (Customer favour)**

#### Examples

- Current account
- Savings account
- Fixed deposit
- Recurring Deposit
- Term Deposits

### NPA

- **Non Performing Asset**
- Loan that is defaulted
- Loan account when DPD > 90 days

### Net Non-Performing Assets (NNPA)

**Definition:** NNPA represents the portion of a bank's non-performing assets (NPAs) that remains after deducting provisions set aside for potential losses.

**Calculation:** NNPA = Gross NPAs - Provisions

**Significance:** NNPA gives a more accurate view of a bank's financial health regarding bad loans because it shows the actual potential losses if borrowers default on their NPAs.

**Example:**
Let's say a bank has the following figures:

- Gross NPAs (total value of NPAs): $10 million
- Provisions (reserve funds for potential losses): $3 million

NNPA = $10 million - $3 million = $7 million

This means the bank has $7 million of NPAs after considering provisions.

### Gross Non-Performing Assets (GNPA)

**Definition:** GNPA represents the total value of a bank's non-performing assets (NPAs) before accounting for any provisions.

**Calculation:** GNPA = Total value of all NPAs (Substandard, Doubtful, and Loss Assets)

**Significance:** GNPA provides a broad picture of a bank's asset quality and the overall level of stressed loans on its books.

**Example:**
Suppose a bank has the following NPAs in different categories:

- Substandard Assets: $5 million
- Doubtful Debts: $3 million
- Loss Assets: $2 million

GNPA = $5 million (Substandard) + $3 million (Doubtful) + $2 million (Loss) = $10 million

This means the bank has $10 million of NPAs before considering any provisions.

### Loan Classification Stages

Let's now explore the stages of loan classification from Days Past Due (DPD) to Write-Off:

1. **DPD (Days Past Due):**

   - **Definition:** DPD refers to the number of days a borrower has exceeded the due date for a loan payment.
   - **Example:** If a borrower's monthly loan payment is due on the 1st of each month and they miss the payment, they enter DPD status. For instance, if the current date is the 15th and the borrower hasn't made the payment, they are 15 days past due.
2. **Non-performing Asset (NPA):**

   - **Definition:** A loan becomes an NPA if the borrower misses payments for a specific period, often 90 days or more.
   - **Example:** If a borrower hasn't made any payments for 90 days on a loan, it is classified as an NPA.
3. **Special Mention Account (SMA):**

   - **Definition:** SMA refers to loans identified by banks as having a high potential of turning into NPAs.
   - **Example:** A loan with consistent delays in payments or signs of financial stress from the borrower may be categorized as SMA.
4. **Write-off:**

   - **Definition:** A loan is written off when the bank deems it unrecoverable and removes it from active accounts.
   - **Example:** If all attempts to recover a loan fail, such as through legal actions or negotiations, the bank may decide to write off the loan as a loss.

Understanding these stages helps banks manage risks associated with loans and take appropriate actions to recover funds or minimize losses.

### Disbursed Amount

- DA is a amount given to a customer as loan

### OSP

- **Out Standing Principle**
- In simple words OSP is the amount that is remaining to return the loan to bank
- OSP should be zero at the end of loan cycle

### DPD

- **Days Past Due**
- **Defaulted** if $DPD > 0$

### PAR

- **Portfolio At Risk**
- OSP when DPD > 0

## Credit Risk Types in Banking

- DPD (Zero) : NDA (Non delinquint account) = No default account = Timely payment EMI
- DPD (0 to 30) : SMA1 (Standard Monitoring Account)
- DPD (31 to 60) : SMA2 (Standard Monitoring Account)
- DPD (61 to 90) : SMA3 (Standard Monitoring Account)
- DPD (90 to 180) : NPA
- DPD (>180) : Writen-off (Loan which is not present)
  **NNPA (Net Non-Performing Assets)**

* **Definition:** This represents the portion of a bank's non-performing assets (NPAs) remaining *after* deducting provisions set aside for potential losses. Provisions are like reserve funds created by banks to anticipate losses on bad loans.
* **Calculation:**
  NNPA = Gross NPAs – Provisions
* **Significance:** NNPA is a more accurate measure of a bank's true financial health with regards to bad loans. It shows the actual potential losses the bank might face if borrowers completely default on their NPAs.

**GNPA (Gross Non-Performing Assets)**

* **Definition:** This represents the total value of a bank's non-performing assets (NPAs) *before* accounting for any provisions.
* **Calculation:**
  GNPA =  Total value of all NPAs (Substandard, Doubtful, and Loss Assets)
* **Significance:** GNPA offers a broader picture of a bank's asset quality and the overall level of stressed loans on its books.

**Markdown Table**

| Metric                             | Definition                               | Calculation                                                  | Significance                                                                                 |
| ---------------------------------- | ---------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| GNPA (Gross Non-Performing Assets) | Total value of NPAs before provisions    | Total value of all NPAs (Substandard, Doubtful, Loss Assets) | Indicates overall level of stressed loans                                                    |
| NNPA (Net Non-Performing Assets)   | Value of NPAs after deducting provisions | Gross NPAs - Provisions                                      | Reveals potential losses if borrowers default, a better indicator of bank's financial health |

**Key Points**

* Banks aim to keep both GNPA and NNPA ratios low.
* High GNPA and NNPA ratios signal potential financial distress for a bank.
* Investors and regulators closely track these metrics to assess a bank's stability.

### NPA Impact

- NPA improve = Loan Portfolio quality of the bank will be better = Market sentiment will be good = Stock price will improve

## Datasets

### Internal Dataset

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Variable Name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Tot_Closed_T</td>
      <td>Total closed trade lines/accounts</td>
    </tr>
    <tr>
      <td>Tot_Active_TL</td>
      <td>Total active accounts</td>
    </tr>
    <tr>
      <td>Total_TL_opened_L6M</td>
      <td>Total accounts opened in last 6 Months</td>
    </tr>
    <tr>
      <td>Tot_TL_closed_L6M</td>
      <td>Total accounts closed in last 6 months</td>
    </tr>
    <tr>
      <td>pct_tl_open_L6M</td>
      <td>Percent accounts opened in last 6 months</td>
    </tr>
    <tr>
      <td>pct_tl_closed_L6M</td>
      <td>percent accounts closed in last 6 months</td>
    </tr>
    <tr>
      <td>pct_active_tl</td>
      <td>Percent active accounts</td>
    </tr>
    <tr>
      <td>pct_closed_tl</td>
      <td>Percent closed accounts</td>
    </tr>
    <tr>
      <td>Total_TL_opened_L12M</td>
      <td>Total accounts opened in last 12 Months</td>
    </tr>
    <tr>
      <td>Tot_TL_closed_L12M</td>
      <td>Total accounts closed in last 12 months</td>
    </tr>
    <tr>
      <td>pct_tl_open_L12M</td>
      <td>Percent accounts opened in last 12 months</td>
    </tr>
    <tr>
      <td>pct_tl_closed_L12M</td>
      <td>percent accounts closed in last 12 months</td>
    </tr>
    <tr>
      <td>Tot Missed Pmnt</td>
      <td>Total missed Payments</td>
    </tr>
    <tr>
      <td>Auto TL</td>
      <td>Count Automobile accounts</td>
    </tr>
    <tr>
      <td>CC TL</td>
      <td>Count of Credit card accounts</td>
    </tr>
    <tr>
      <td>Consumer_TL</td>
      <td>Count of Consumer goods accounts</td>
    </tr>
    <tr>
      <td>Gold TL</td>
      <td>Count of Gold loan accounts</td>
    </tr>
    <tr>
      <td>Home_TL</td>
      <td>Count of Housing loan accounts</td>
    </tr>
    <tr>
      <td>PLOTL</td>
      <td>Count of Personal loan accounts</td>
    </tr>
    <tr>
      <td>Secured_TL</td>
      <td>Count of secured accounts</td>
    </tr>
    <tr>
      <td>Unsecured_TL</td>
      <td>Count of unsecured accounts</td>
    </tr>
    <tr>
      <td>Other_TL</td>
      <td>Count of other accounts</td>
    </tr>
    <tr>
      <td>Age_Oldest_TL</td>
      <td>Age of oldest opened account</td>
    </tr>
    <tr>
      <td>Age_Newest_TL</td>
      <td>Age of newest opened account</td>
    </tr>
  </tbody>
</table>

### External Dataset (Cibil)

<table border="1">
  <thead>
    <tr>
      <th>#</th>
      <th>Variable Name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>time_since_recent_payment</td>
      <td>Time Since recent Payment made</td>
    </tr>
    <tr>
      <td>2</td>
      <td>time_since_first_deliquency</td>
      <td>Time since first Deliquency (missed payment)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>time_since_recent_deliquency</td>
      <td>Time Since recent Delinquency</td>
    </tr>
    <tr>
      <td>4</td>
      <td>num_times_delinquent</td>
      <td>Number of times delinquent</td>
    </tr>
    <tr>
      <td>5</td>
      <td>max_delinquency_level</td>
      <td>Maximum delinquency level</td>
    </tr>
    <tr>
      <td>6</td>
      <td>max_recent_level_of_deliq</td>
      <td>Maximum recent level of delinquency</td>
    </tr>
    <tr>
      <td>7</td>
      <td>num_deliq_6mts</td>
      <td>Number of times delinquent in last 6 months</td>
    </tr>
    <tr>
      <td>8</td>
      <td>num_deliq_12mts</td>
      <td>Number of times delinquent in last 12 months</td>
    </tr>
    <tr>
      <td>9</td>
      <td>num_deliq_6_12mts</td>
      <td>Number of times delinquent between last 6 months and last 12 months</td>
    </tr>
    <tr>
      <td>10</td>
      <td>max_deliq_6mts</td>
      <td>Maximum delinquency level in last 6 months</td>
    </tr>
    <tr>
      <td>11</td>
      <td>max_deliq_12mts</td>
      <td>Maximum delinquency level in last 12 months</td>
    </tr>
    <tr>
      <td>12</td>
      <td>num_times_30p_dpd</td>
      <td>Number of times 30+ dpd</td>
    </tr>
    <tr>
      <td>13</td>
      <td>num_times_60p_dpd</td>
      <td>Number of times 60+ dpd</td>
    </tr>
    <tr>
      <td>14</td>
      <td>num_std</td>
      <td>Number of standard Payments</td>
    </tr>
    <tr>
      <td>15</td>
      <td>num_std_6mts</td>
      <td>Number of standard Payments in last 6 months</td>
    </tr>
    <tr>
      <td>16</td>
      <td>num_std_12mts</td>
      <td>Number of standard Payments in last 12 months</td>
    </tr>
    <tr>
      <td>17</td>
      <td>num_sub</td>
      <td>Number of sub standard payments - not making full payments</td>
    </tr>
    <tr>
      <td>18</td>
      <td>num_sub_6mts</td>
      <td>Number of sub standard payments in last 6 months</td>
    </tr>
    <tr>
      <td>19</td>
      <td>num_sub_12mts</td>
      <td>Number of sub standard payments in last 12 months</td>
    </tr>
    <tr>
      <td>20</td>
      <td>num_dbt</td>
      <td>Number of doubtful payments</td>
    </tr>
    <tr>
      <td>21</td>
      <td>num_dbt_6mts</td>
      <td>Number of doubtful payments in last 6 months</td>
    </tr>
    <tr>
      <td>22</td>
      <td>num_dbt_12mts</td>
      <td>Number of doubtful payments in last 12 months</td>
    </tr>
    <tr>
      <td>23</td>
      <td>num_Iss</td>
<td>Number of loss accounts</td>
</tr>
<tr>
<td>24</td>
<td>num_lss_6mts</td>
<td>Number of loss accounts in last 6 months</td>
</tr>
<tr>
<td>25</td>
<td>num_lss_12mts</td>
<td>Number of loss accounts in last 12 months</td>
</tr>
<tr>
<td>26</td>
<td>recent_level_of_deliq</td>
<td>Recent level of delinquency</td>
</tr>
<tr>
<td>27</td>
<td>tot_enq</td>
<td>Total enquiries</td>
</tr>
<tr>
<td>28</td>
<td>CC_enq</td>
<td>Credit card enquiries</td>
</tr>
<tr>
<td>29</td>
<td>CC_enq_L6m</td>
<td>Credit card enquiries in last 6 months</td>
</tr>
<tr>
<td>30</td>
<td>CC_enq_L12m</td>
<td>Credit card enquiries in last 12 months</td>
</tr>
<tr>
<td>31</td>
<td>PL_enq</td>
<td>Personal Loan enquiries</td>
</tr>
<tr>
<td>32</td>
<td>PL_enq_L6m</td>
<td>Personal Loan enquiries in last 6 months</td>
</tr>
<tr>
<td>33</td>
<td>PL_enq_L12m</td>
<td>Personal Loan enquiries in last 12 months</td>
</tr>
<tr>
<td>34</td>
<td>time_since_recent_enq</td>
<td>Time since recent enquiry</td>
</tr>
<tr>
<td>35</td>
<td>enq_L12m</td>
<td>Enquiries in last 12 months</td>
</tr>
<tr>
<td>36</td>
<td>enq_L6m</td>
<td>Enquiries in last 6 months</td>
</tr>
<tr>
<td>37</td>
<td>enq_L3m</td>
<td>Enquiries in last 3 months</td>
</tr>
<tr>
<td>38</td>
<td>MARITALSTATUS</td>
<td>Marital Status</td>
</tr>
<tr>
<td>39</td>
<td>EDUCATION</td>
<td>Education level</td>
</tr>
<tr>
<td>AGE</td>
<td>Age</td>
</tr>
<tr>
<td>41</td>
<td>GENDER</td>
<td>Gender</td>
</tr>
  <tr>
      <th>42</th>
      <td>NETMONTHLYINCOME</td>
      <td>Net monthly income</td>
    </tr>
      <tr>
      <th>43</th>
      <td>Time_With_Curr_Empr</td>
      <td>Time with current Employer</td>
    </tr>
    <tr>
      <th>44</th>
      <td>pct_of_active_TLs_ever</td>
      <td>Percent active accounts ever</td>
    </tr>
    <tr>
      <th>45</th>
      <td>pct_opened_TLs_L6m_of_L12m</td>
      <td>Percent accounts opened in last 6 months to last 12 months</td>
    </tr>
    <tr>
      <th>46</th>
      <td>pct_currentBal_all_TL</td>
      <td>Percent current balance of all accounts</td>
    </tr>
    <tr>
      <th>47</th>
      <td>CC_utilization</td>
      <td>Credit card utilization</td>
    </tr>
    <tr>
      <th>48</th>
      <td>CC_Flag</td>
      <td>Credit card Flag</td>
    </tr>
    <tr>
      <th>49</th>
      <td>PL_utilization</td>
      <td>Peronal Loan utilization</td>
    </tr>
    <tr>
      <th>50</th>
      <td>PL_Flag</td>
      <td>Personal Loan Flag</td>
    </tr>
    <tr>
      <th>51</th>
      <td>pct_PL_enq_L6m_of_L12m</td>
      <td>Percent enquiries PL in last 6 months to last 12 months</td>
    </tr>
    <tr>
      <th>52</th>
      <td>pct_CC_enq_L6m_of_L12m</td>
      <td>Percent enquiries CC in last 6 months to last 12 months</td>
    </tr>
    <tr>
      <th>53</th>
      <td>pct_PL_enq_L6m_of_ever</td>
      <td>Percent enquiries PL in last 6 months to last 6 months</td>
    </tr>
    <tr>
      <th>54</th>
      <td>pct_CC_enq_L6m_of_ever</td>
      <td>Percent enquiries CC in last 6 months to last 6 months</td>
    </tr>
    <tr>
      <th>55</th>
      <td>max_unsec_exposure_inPct</td>
      <td>Maximum unsecured exposure in percent</td>
    </tr>
    <tr>
      <th>56</th>
      <td>HL_Flag</td>
      <td>Housing Loan Flag</td>
    </tr>
    <tr>
      <th>57</th>
      <td>GL_Flag</td>
      <td>Gold Loan Flag</td>
    </tr>
    <tr>
      <th>58</th>
      <td>last_prod_enq2</td>
      <td>Lates product enquired for</td>
    </tr>
    <tr>
      <th>59</th>
      <td>first_prod_enq2</td>
      <td>First product enquired for</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Credit_Score</td>
      <td>Applicant's credit score</td>
    </tr>
    <tr>
      <th>61</th>
      <td>Approved_Flag</td>
      <td>Priority levels</td>
    </tr>
</tbody>
</table>

Accuracy :- Out of total values,how many are correctly predicted

Recall :- Out of my total value for a class how many are correctly predicted

Precision :-  Out of my total predicted value for a class how many are correctly predicted

F1-Score :- 2*P*R/(P+R)

## Banking Termninoligies

### Asset

- **Asset is something that can give some profit to bank finally. (Bank favour)**
- Known as Loan Product

#### Examples

- Housing loan
- Personal loan
- Vehicle loan
- Group loan
- Education loan
- Credit Card

### Liability

- **Liability is something that can give some Loss to bank finally. (Customer favour)**

#### Examples

- Current account
- Savings account
- Fixed deposit
- Recurring Deposit
- Term Deposits

### NPA

- **Non Performing Asset**
- Loan that is defaulted
- Loan account when DPD > 90 days

### Net Non-Performing Assets (NNPA)

**Definition:** NNPA represents the portion of a bank's non-performing assets (NPAs) that remains after deducting provisions set aside for potential losses.

**Calculation:** NNPA = Gross NPAs - Provisions

**Significance:** NNPA gives a more accurate view of a bank's financial health regarding bad loans because it shows the actual potential losses if borrowers default on their NPAs.

**Example:**
Let's say a bank has the following figures:

- Gross NPAs (total value of NPAs): $10 million
- Provisions (reserve funds for potential losses): $3 million

NNPA = $10 million - $3 million = $7 million

This means the bank has $7 million of NPAs after considering provisions.

### Gross Non-Performing Assets (GNPA)

**Definition:** GNPA represents the total value of a bank's non-performing assets (NPAs) before accounting for any provisions.

**Calculation:** GNPA = Total value of all NPAs (Substandard, Doubtful, and Loss Assets)

**Significance:** GNPA provides a broad picture of a bank's asset quality and the overall level of stressed loans on its books.

**Example:**
Suppose a bank has the following NPAs in different categories:

- Substandard Assets: $5 million
- Doubtful Debts: $3 million
- Loss Assets: $2 million

GNPA = $5 million (Substandard) + $3 million (Doubtful) + $2 million (Loss) = $10 million

This means the bank has $10 million of NPAs before considering any provisions.

### Loan Classification Stages

Let's now explore the stages of loan classification from Days Past Due (DPD) to Write-Off:

1. **DPD (Days Past Due):**

   - **Definition:** DPD refers to the number of days a borrower has exceeded the due date for a loan payment.
   - **Example:** If a borrower's monthly loan payment is due on the 1st of each month and they miss the payment, they enter DPD status. For instance, if the current date is the 15th and the borrower hasn't made the payment, they are 15 days past due.
2. **Non-performing Asset (NPA):**

   - **Definition:** A loan becomes an NPA if the borrower misses payments for a specific period, often 90 days or more.
   - **Example:** If a borrower hasn't made any payments for 90 days on a loan, it is classified as an NPA.
3. **Special Mention Account (SMA):**

   - **Definition:** SMA refers to loans identified by banks as having a high potential of turning into NPAs.
   - **Example:** A loan with consistent delays in payments or signs of financial stress from the borrower may be categorized as SMA.
4. **Write-off:**

   - **Definition:** A loan is written off when the bank deems it unrecoverable and removes it from active accounts.
   - **Example:** If all attempts to recover a loan fail, such as through legal actions or negotiations, the bank may decide to write off the loan as a loss.

Understanding these stages helps banks manage risks associated with loans and take appropriate actions to recover funds or minimize losses.

### Disbursed Amount

- DA is a amount given to a customer as loan

### OSP

- **Out Standing Principle**
- In simple words OSP is the amount that is remaining to return the loan to bank
- OSP should be zero at the end of loan cycle

### DPD

- **Days Past Due**
- **Defaulted** if $DPD > 0$

### PAR

- **Portfolio At Risk**
- OSP when DPD > 0

## Credit Risk Types in Banking

- DPD (Zero) : NDA (Non delinquint account) = No default account = Timely payment EMI
- DPD (0 to 30) : SMA1 (Standard Monitoring Account)
- DPD (31 to 60) : SMA2 (Standard Monitoring Account)
- DPD (61 to 90) : SMA3 (Standard Monitoring Account)
- DPD (90 to 180) : NPA
- DPD (>180) : Writen-off (Loan which is not present)
  **NNPA (Net Non-Performing Assets)**

* **Definition:** This represents the portion of a bank's non-performing assets (NPAs) remaining *after* deducting provisions set aside for potential losses. Provisions are like reserve funds created by banks to anticipate losses on bad loans.
* **Calculation:**
  NNPA = Gross NPAs – Provisions
* **Significance:** NNPA is a more accurate measure of a bank's true financial health with regards to bad loans. It shows the actual potential losses the bank might face if borrowers completely default on their NPAs.

**GNPA (Gross Non-Performing Assets)**

* **Definition:** This represents the total value of a bank's non-performing assets (NPAs) *before* accounting for any provisions.
* **Calculation:**
  GNPA =  Total value of all NPAs (Substandard, Doubtful, and Loss Assets)
* **Significance:** GNPA offers a broader picture of a bank's asset quality and the overall level of stressed loans on its books.

**Markdown Table**

| Metric                             | Definition                               | Calculation                                                  | Significance                                                                                 |
| ---------------------------------- | ---------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| GNPA (Gross Non-Performing Assets) | Total value of NPAs before provisions    | Total value of all NPAs (Substandard, Doubtful, Loss Assets) | Indicates overall level of stressed loans                                                    |
| NNPA (Net Non-Performing Assets)   | Value of NPAs after deducting provisions | Gross NPAs - Provisions                                      | Reveals potential losses if borrowers default, a better indicator of bank's financial health |

**Key Points**

* Banks aim to keep both GNPA and NNPA ratios low.
* High GNPA and NNPA ratios signal potential financial distress for a bank.
* Investors and regulators closely track these metrics to assess a bank's stability.

### NPA Impact

- NPA improve = Loan Portfolio quality of the bank will be better = Market sentiment will be good = Stock price will improve

## Datasets

### Internal Dataset

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Variable Name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Tot_Closed_T</td>
      <td>Total closed trade lines/accounts</td>
    </tr>
    <tr>
      <td>Tot_Active_TL</td>
      <td>Total active accounts</td>
    </tr>
    <tr>
      <td>Total_TL_opened_L6M</td>
      <td>Total accounts opened in last 6 Months</td>
    </tr>
    <tr>
      <td>Tot_TL_closed_L6M</td>
      <td>Total accounts closed in last 6 months</td>
    </tr>
    <tr>
      <td>pct_tl_open_L6M</td>
      <td>Percent accounts opened in last 6 months</td>
    </tr>
    <tr>
      <td>pct_tl_closed_L6M</td>
      <td>percent accounts closed in last 6 months</td>
    </tr>
    <tr>
      <td>pct_active_tl</td>
      <td>Percent active accounts</td>
    </tr>
    <tr>
      <td>pct_closed_tl</td>
      <td>Percent closed accounts</td>
    </tr>
    <tr>
      <td>Total_TL_opened_L12M</td>
      <td>Total accounts opened in last 12 Months</td>
    </tr>
    <tr>
      <td>Tot_TL_closed_L12M</td>
      <td>Total accounts closed in last 12 months</td>
    </tr>
    <tr>
      <td>pct_tl_open_L12M</td>
      <td>Percent accounts opened in last 12 months</td>
    </tr>
    <tr>
      <td>pct_tl_closed_L12M</td>
      <td>percent accounts closed in last 12 months</td>
    </tr>
    <tr>
      <td>Tot Missed Pmnt</td>
      <td>Total missed Payments</td>
    </tr>
    <tr>
      <td>Auto TL</td>
      <td>Count Automobile accounts</td>
    </tr>
    <tr>
      <td>CC TL</td>
      <td>Count of Credit card accounts</td>
    </tr>
    <tr>
      <td>Consumer_TL</td>
      <td>Count of Consumer goods accounts</td>
    </tr>
    <tr>
      <td>Gold TL</td>
      <td>Count of Gold loan accounts</td>
    </tr>
    <tr>
      <td>Home_TL</td>
      <td>Count of Housing loan accounts</td>
    </tr>
    <tr>
      <td>PLOTL</td>
      <td>Count of Personal loan accounts</td>
    </tr>
    <tr>
      <td>Secured_TL</td>
      <td>Count of secured accounts</td>
    </tr>
    <tr>
      <td>Unsecured_TL</td>
      <td>Count of unsecured accounts</td>
    </tr>
    <tr>
      <td>Other_TL</td>
      <td>Count of other accounts</td>
    </tr>
    <tr>
      <td>Age_Oldest_TL</td>
      <td>Age of oldest opened account</td>
    </tr>
    <tr>
      <td>Age_Newest_TL</td>
      <td>Age of newest opened account</td>
    </tr>
  </tbody>
</table>

### External Dataset (Cibil)

<table border="1">
  <thead>
    <tr>
      <th>#</th>
      <th>Variable Name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>time_since_recent_payment</td>
      <td>Time Since recent Payment made</td>
    </tr>
    <tr>
      <td>2</td>
      <td>time_since_first_deliquency</td>
      <td>Time since first Deliquency (missed payment)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>time_since_recent_deliquency</td>
      <td>Time Since recent Delinquency</td>
    </tr>
    <tr>
      <td>4</td>
      <td>num_times_delinquent</td>
      <td>Number of times delinquent</td>
    </tr>
    <tr>
      <td>5</td>
      <td>max_delinquency_level</td>
      <td>Maximum delinquency level</td>
    </tr>
    <tr>
      <td>6</td>
      <td>max_recent_level_of_deliq</td>
      <td>Maximum recent level of delinquency</td>
    </tr>
    <tr>
      <td>7</td>
      <td>num_deliq_6mts</td>
      <td>Number of times delinquent in last 6 months</td>
    </tr>
    <tr>
      <td>8</td>
      <td>num_deliq_12mts</td>
      <td>Number of times delinquent in last 12 months</td>
    </tr>
    <tr>
      <td>9</td>
      <td>num_deliq_6_12mts</td>
      <td>Number of times delinquent between last 6 months and last 12 months</td>
    </tr>
    <tr>
      <td>10</td>
      <td>max_deliq_6mts</td>
      <td>Maximum delinquency level in last 6 months</td>
    </tr>
    <tr>
      <td>11</td>
      <td>max_deliq_12mts</td>
      <td>Maximum delinquency level in last 12 months</td>
    </tr>
    <tr>
      <td>12</td>
      <td>num_times_30p_dpd</td>
      <td>Number of times 30+ dpd</td>
    </tr>
    <tr>
      <td>13</td>
      <td>num_times_60p_dpd</td>
      <td>Number of times 60+ dpd</td>
    </tr>
    <tr>
      <td>14</td>
      <td>num_std</td>
      <td>Number of standard Payments</td>
    </tr>
    <tr>
      <td>15</td>
      <td>num_std_6mts</td>
      <td>Number of standard Payments in last 6 months</td>
    </tr>
    <tr>
      <td>16</td>
      <td>num_std_12mts</td>
      <td>Number of standard Payments in last 12 months</td>
    </tr>
    <tr>
      <td>17</td>
      <td>num_sub</td>
      <td>Number of sub standard payments - not making full payments</td>
    </tr>
    <tr>
      <td>18</td>
      <td>num_sub_6mts</td>
      <td>Number of sub standard payments in last 6 months</td>
    </tr>
    <tr>
      <td>19</td>
      <td>num_sub_12mts</td>
      <td>Number of sub standard payments in last 12 months</td>
    </tr>
    <tr>
      <td>20</td>
      <td>num_dbt</td>
      <td>Number of doubtful payments</td>
    </tr>
    <tr>
      <td>21</td>
      <td>num_dbt_6mts</td>
      <td>Number of doubtful payments in last 6 months</td>
    </tr>
    <tr>
      <td>22</td>
      <td>num_dbt_12mts</td>
      <td>Number of doubtful payments in last 12 months</td>
    </tr>
    <tr>
      <td>23</td>
      <td>num_Iss</td>
<td>Number of loss accounts</td>
</tr>
<tr>
<td>24</td>
<td>num_lss_6mts</td>
<td>Number of loss accounts in last 6 months</td>
</tr>
<tr>
<td>25</td>
<td>num_lss_12mts</td>
<td>Number of loss accounts in last 12 months</td>
</tr>
<tr>
<td>26</td>
<td>recent_level_of_deliq</td>
<td>Recent level of delinquency</td>
</tr>
<tr>
<td>27</td>
<td>tot_enq</td>
<td>Total enquiries</td>
</tr>
<tr>
<td>28</td>
<td>CC_enq</td>
<td>Credit card enquiries</td>
</tr>
<tr>
<td>29</td>
<td>CC_enq_L6m</td>
<td>Credit card enquiries in last 6 months</td>
</tr>
<tr>
<td>30</td>
<td>CC_enq_L12m</td>
<td>Credit card enquiries in last 12 months</td>
</tr>
<tr>
<td>31</td>
<td>PL_enq</td>
<td>Personal Loan enquiries</td>
</tr>
<tr>
<td>32</td>
<td>PL_enq_L6m</td>
<td>Personal Loan enquiries in last 6 months</td>
</tr>
<tr>
<td>33</td>
<td>PL_enq_L12m</td>
<td>Personal Loan enquiries in last 12 months</td>
</tr>
<tr>
<td>34</td>
<td>time_since_recent_enq</td>
<td>Time since recent enquiry</td>
</tr>
<tr>
<td>35</td>
<td>enq_L12m</td>
<td>Enquiries in last 12 months</td>
</tr>
<tr>
<td>36</td>
<td>enq_L6m</td>
<td>Enquiries in last 6 months</td>
</tr>
<tr>
<td>37</td>
<td>enq_L3m</td>
<td>Enquiries in last 3 months</td>
</tr>
<tr>
<td>38</td>
<td>MARITALSTATUS</td>
<td>Marital Status</td>
</tr>
<tr>
<td>39</td>
<td>EDUCATION</td>
<td>Education level</td>
</tr>
<tr>
<td>AGE</td>
<td>Age</td>
</tr>
<tr>
<td>41</td>
<td>GENDER</td>
<td>Gender</td>
</tr>
  <tr>
      <th>42</th>
      <td>NETMONTHLYINCOME</td>
      <td>Net monthly income</td>
    </tr>
      <tr>
      <th>43</th>
      <td>Time_With_Curr_Empr</td>
      <td>Time with current Employer</td>
    </tr>
    <tr>
      <th>44</th>
      <td>pct_of_active_TLs_ever</td>
      <td>Percent active accounts ever</td>
    </tr>
    <tr>
      <th>45</th>
      <td>pct_opened_TLs_L6m_of_L12m</td>
      <td>Percent accounts opened in last 6 months to last 12 months</td>
    </tr>
    <tr>
      <th>46</th>
      <td>pct_currentBal_all_TL</td>
      <td>Percent current balance of all accounts</td>
    </tr>
    <tr>
      <th>47</th>
      <td>CC_utilization</td>
      <td>Credit card utilization</td>
    </tr>
    <tr>
      <th>48</th>
      <td>CC_Flag</td>
      <td>Credit card Flag</td>
    </tr>
    <tr>
      <th>49</th>
      <td>PL_utilization</td>
      <td>Peronal Loan utilization</td>
    </tr>
    <tr>
      <th>50</th>
      <td>PL_Flag</td>
      <td>Personal Loan Flag</td>
    </tr>
    <tr>
      <th>51</th>
      <td>pct_PL_enq_L6m_of_L12m</td>
      <td>Percent enquiries PL in last 6 months to last 12 months</td>
    </tr>
    <tr>
      <th>52</th>
      <td>pct_CC_enq_L6m_of_L12m</td>
      <td>Percent enquiries CC in last 6 months to last 12 months</td>
    </tr>
    <tr>
      <th>53</th>
      <td>pct_PL_enq_L6m_of_ever</td>
      <td>Percent enquiries PL in last 6 months to last 6 months</td>
    </tr>
    <tr>
      <th>54</th>
      <td>pct_CC_enq_L6m_of_ever</td>
      <td>Percent enquiries CC in last 6 months to last 6 months</td>
    </tr>
    <tr>
      <th>55</th>
      <td>max_unsec_exposure_inPct</td>
      <td>Maximum unsecured exposure in percent</td>
    </tr>
    <tr>
      <th>56</th>
      <td>HL_Flag</td>
      <td>Housing Loan Flag</td>
    </tr>
    <tr>
      <th>57</th>
      <td>GL_Flag</td>
      <td>Gold Loan Flag</td>
    </tr>
    <tr>
      <th>58</th>
      <td>last_prod_enq2</td>
      <td>Lates product enquired for</td>
    </tr>
    <tr>
      <th>59</th>
      <td>first_prod_enq2</td>
      <td>First product enquired for</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Credit_Score</td>
      <td>Applicant's credit score</td>
    </tr>
    <tr>
      <th>61</th>
      <td>Approved_Flag</td>
      <td>Priority levels</td>
    </tr>
</tbody>
</table>

Accuracy :- Out of total values,how many are correctly predicted

Recall :- Out of my total value for a class how many are correctly predicted

Precision :-  Out of my total predicted value for a class how many are correctly predicted

F1-Score :- 2*P*R/(P+R)
