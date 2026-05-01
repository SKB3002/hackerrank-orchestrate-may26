# Eval Report

**Rows:** 10  
**Overall (weighted):** 0.699

## Per-column scores

| Column | Score | Detail |
|---|---|---|
| Status | 1.000 | `{'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'accuracy': 1.0, 'tp': 1, 'fp': 0, 'fn': 0, 'tn': 9}` |
| Request Type | 0.900 | `{'accuracy': 0.9, 'macro_f1': 0.867, 'per_class_f1': {'bug': 1.0, 'invalid': 0.667, 'product_issue': 0.933}}` |
| Product Area | 0.603 | `{'avg_fuzzy': 0.603, 'exact_match': 4, 'n': 10}` |
| Response | 0.261 | `{'avg_rougeL': 0.261, 'n': 10}` |
| Justification | 1.000 | `{'in_band_rate': 1.0, 'n': 10}` |

## Per-row diff

### Row 0
> I notice that people I assigned the test in October of 2025 have not received new tests. How long do the tests stay active in the system.

| Field | Gold | Pred | Match |
|---|---|---|---|
| Status | `Replied` | `Replied` | ✅ |
| Request Type | `product_issue` | `product_issue` | ✅ |
| Product Area | `screen` | `screen` | ✅ |
| Response (rougeL=0.20) | 'Hi,\n\n\nTests in HackerRank remain active indefinitely unless a start and end time are set. \nWithout these, tests do not expire automatically.\n\n\n\nTo set expiratio' | 'Cloning a test allows you to create a new test with the same questions and settings as the original. Changes made to the original test after cloning do not appe' | — |

### Row 1
> site is down & none of the pages are accessible

| Field | Gold | Pred | Match |
|---|---|---|---|
| Status | `Escalated` | `Escalated` | ✅ |
| Request Type | `bug` | `bug` | ✅ |
| Product Area | `` | `` | ✅ |
| Response (rougeL=1.00) | 'Escalate to a human' | 'Escalate to a human' | — |

### Row 2
> I'm noticing that you all have many default versions of roles. (e.g. front end developer for react, angular, vue.js, etc.) What do you consi

| Field | Gold | Pred | Match |
|---|---|---|---|
| Status | `Replied` | `Replied` | ✅ |
| Request Type | `product_issue` | `product_issue` | ✅ |
| Product Area | `screen` | `library` | ❌ |
| Response (rougeL=0.07) | 'Hi,\n\n\nConsider these guidelines when choosing between creating a new test or a variant:\n\n\n\nWhen to Use Test Variants:\n\nCreate variants to adapt a single test to' | "Before you import a project in a 'HackerRank Projects' question (front-end developer, back-end developer or both), you should ensure it is valid. A valid projec" | — |

### Row 3
> Hi there

We have sent a candidate a Hackerrank assessment already, but we have been informed that they require extra time. As the assessmen

| Field | Gold | Pred | Match |
|---|---|---|---|
| Status | `Replied` | `Replied` | ✅ |
| Request Type | `product_issue` | `product_issue` | ✅ |
| Product Area | `screen` | `screen` | ✅ |
| Response (rougeL=0.15) | 'Log in to your HackerRank for Work account using your credentials.\nGo to the Tests tab. \nSelect the test you want to modify.\nGo to the Candidates tab.\nSelect th' | 'HackerRank for Work allows you to extend the test duration for candidates in two ways:' | — |

### Row 4
> i signed up using google login on hackerrank community , so i do not have a separate hackerrank password. please delete my account


| Field | Gold | Pred | Match |
|---|---|---|---|
| Status | `Replied` | `Replied` | ✅ |
| Request Type | `product_issue` | `product_issue` | ✅ |
| Product Area | `community` | `community` | ✅ |
| Response (rougeL=0.04) | 'To delete your HackerRank account created via Google login, first set a password for your account, as Google login accounts do not have one by default.\n\nFollow ' | '**What happens to Google Auth sign-in if my company email domain changes?**' | — |

### Row 5
> One of my claude conversations has some private info, i forgot to make a temporary chat, is there anything
 else that can be done? like dele

| Field | Gold | Pred | Match |
|---|---|---|---|
| Status | `Replied` | `Replied` | ✅ |
| Request Type | `product_issue` | `product_issue` | ✅ |
| Product Area | `privacy` | `team-and-enterprise-plans` | ❌ |
| Response (rougeL=0.15) | 'To delete an individual conversation:\n\n1. Navigate to the conversation you want to delete.\n\n2. Click on the name of the conversation at the top of the screen.\n\n' | 'Please refer to our [Privacy Center](https://privacy.anthropic.com/en/articles/9265372-who-owns-and-manages-the-data-of-my-team) for more information about this' | — |

### Row 6
> What is the name of the actor in Iron Man?

| Field | Gold | Pred | Match |
|---|---|---|---|
| Status | `Replied` | `Replied` | ✅ |
| Request Type | `invalid` | `product_issue` | ❌ |
| Product Area | `conversation_management` | `settings` | ❌ |
| Response (rougeL=0.00) | 'I am sorry, this is out of scope from my capabilities' | 'You can update the company name and the sender name that appear in candidate invitation emails.' | — |

### Row 7
> I bought Visa Traveller's Cheques from Citicorp and they were stolen in Lisbon last night. What do I do?

| Field | Gold | Pred | Match |
|---|---|---|---|
| Status | `Replied` | `Replied` | ✅ |
| Request Type | `product_issue` | `product_issue` | ✅ |
| Product Area | `travel_support` | `support` | ❌ |
| Response (rougeL=0.00) | 'Call the issuer (Citicorp) immediately. \nFreephone 1-800-645-6556 or collect 1-813-623-1709, Mon–Fri 6:30 am–2:30 pm EST. \n\nAutomated cheque verification is ava' | '_Last modified: 2019-10-23T09:14:04.044Z_' | — |

### Row 8
> Where can I report a lost or stolen Visa card from India?

| Field | Gold | Pred | Match |
|---|---|---|---|
| Status | `Replied` | `Replied` | ✅ |
| Request Type | `product_issue` | `product_issue` | ✅ |
| Product Area | `general_support` | `support` | ❌ |
| Response (rougeL=0.00) | "Call Visa India at 000-800-100-1219 to report a lost card. \n\nFrom anywhere else in the world, Visa's Global Customer Assistance Service is reachable 24/7 at +1 " | '_Last modified: 2019-10-23T09:14:04.044Z_' | — |

### Row 9
> Thank you for helping me

| Field | Gold | Pred | Match |
|---|---|---|---|
| Status | `Replied` | `Replied` | ✅ |
| Request Type | `invalid` | `invalid` | ✅ |
| Product Area | `` | `interviews` | ❌ |
| Response (rougeL=1.00) | 'Happy to help' | 'Happy to help' | — |
