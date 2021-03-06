---
layout: default
---

## Persian Category Detection

In summer 2017 I was working on a project in [*Yektanet*](http://yektanet.com/) which is one of the biggest online advertising platforms in Iran.
Result of this project was an __automated system for recognizing text content and categorizing persian web pages__.
We had analyzed our publishers and based on their contents we defined 25 different categories.
These categories are including 
1. Real Estate
2. Car
3. Food
4. Tourism
5. Literature
6. Fitness and Health
7. Fun
8. HouseKeeping and Gardening
9. Sport
10. Animals and Pet
11. Politics and Social
12. Employment
13. Finance
14. Entertainment
15. Information Technology
16. Mobile 
17. Computer
18. Art and Media
19. Fashion
20. Bank and Insurance
21. Internet Service Providers
22. Home Appliances
23. Kids and Teenagers
24. Marriage
25. Games

We predefined category of enormous web pages and crawled them to gather learning data.
By crawling these pages we gathered a word-category co-occurrence table. Using this table we normalized co-occurrence of word to get a distribution on different categories for each word. With calculating entropy of each word and sorting according to entropy we find words which does'nt have any role in defining category.

For categorizing a new text we use it's words and naive Bayes assumption and we multiply score of each word by a gaussian function of entropy to reduce effect of uncertain words.
We achieved accuracy of 0.95% on our test data. A demo of our project is available on [Yektanet Website](https://yektanet.com/%D8%AA%D8%B4%D8%AE%DB%8C%D8%B5-%D8%AE%D9%88%D8%AF%DA%A9%D8%A7%D8%B1-%D9%85%D9%88%D8%B6%D9%88%D8%B9/).

[back](./)
