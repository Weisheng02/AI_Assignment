# TAR UMT answer sources

Verified against TAR UMT-controlled pages on 24 August 2026. These links are
the primary sources used by `data/intents.json`; the chatbot deliberately
avoids storing unsupported fee ranges, intake months, employment percentages,
room details, menus, office holders, or opening hours as timeless facts.

| Topic | Official primary source |
|---|---|
| University and campus contacts | https://www.tarc.edu.my/contact-us/ |
| Admissions contact and opening hours | https://www.tarc.edu.my/admissions/contact-us/ |
| Undergraduate programme directory | https://www.tarc.edu.my/admissions/programmes/programme-offered-a-z/undergraduate-programme/ |
| Malaysian student fees | https://www.tarc.edu.my/bursary/malaysian-student-fees-guide/ |
| International student fees | https://www.tarc.edu.my/intfees/international-student-fees-guide-for-year-2026-intakes/ |
| Current intake notices | https://dace.tarc.edu.my/programmes/intakes |
| Academic calendars | https://www.tarc.edu.my/admissions/new-student/academic-calendar/ |
| Examination and assessment-result FAQ | https://examination.tarc.edu.my/examination-services/faqs |
| Application, documents and enrolment | https://www.tarc.edu.my/admissions/a/application-and-enrolment-status-enquiry/ |
| Admissions FAQ | https://www.tarc.edu.my/admissions/faqs/ |
| KL accommodation | https://www.tarc.edu.my/dsa/a/accommodation/accommodation-kl-main-campus/ |
| Library opening hours | https://library.tarc.edu.my/about-us/opening-hours |
| Library facilities | https://library.tarc.edu.my/services-facilities/facilities |
| Food and beverage outlets | https://www.tarc.edu.my/dsa/food-and-beverage/ |
| University bus service | https://www.tarc.edu.my/dsa/a/transportation/university-college-bus-service/ |
| Sports facilities | https://www.tarc.edu.my/dsa/contentsub.jsp?cat_id=5A2E4500-18D3-48FA-83FE-1374A4FA6C74&fmenuid=57844F48-0FC8-412E-A113-8FA4CA6FBC17 |
| Dress code | https://www.tarc.edu.my/dsa/contentsub.jsp?cat_id=3F86463C-2727-456C-AC7D-44A833E590A6&fmenuid= |
| Clubs and societies | https://www.tarc.edu.my/dsa/contentsub.jsp?cat_id=BB59517D-5141-4726-A32B-4293FE854DEB&fmenuid=424F47AF-637B-44EC-AFB3-3B4DF0861DAC |
| Scholarships and grants | https://www.tarc.edu.my/dsa/financial-aid/scholarships-grants/ |
| Student Career Development Centre | https://scdcstu.tarc.edu.my/ |
| President page | https://www.tarc.edu.my/tarc-umt/president-welcome-message/ |
| Staff directory | https://www.tarc.edu.my/staffDirectory.jsp |
| University bulletin | https://www.tarc.edu.my/bulletin.jsp |
| Student Code of Conduct | https://www.tarc.edu.my/files/dsa/CD2F46FC-2B88-4ED6-97F2-4F30CB6E4052.pdf |
| Kuala Lumpur campus overview | https://www.tarc.edu.my/kl/index.jsp |

## Response policy

- A stable address may be stated when the same official response also links to
  the live directory.
- Volatile amounts, dates, availability, staff names, operating hours and
  eligibility rules are not cached in a generic answer. The user is sent to the
  topic-specific official page.
- The bot identifies itself as a student prototype, not an official TAR UMT
  service.
- A webhook may later retrieve official live directory/programme data, but no
  live lookup is claimed until an HTTPS endpoint is configured and tested.
