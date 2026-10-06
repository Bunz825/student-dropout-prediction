# Data Dictionary

All 37 columns, in file order. Observed ranges come from the project data; code meanings come from the creators' documentation (UCI page and Realinho et al., 2022). 

#	Variable (exact column name)	Type	Unit / scale	Allowed / observed values	Description
1	Marital Status	Nominal	Code	1–6 (6 codes)	Marital status at enrolment: 1 single, 2 married, 3 widower, 4 divorced, 5 facto union, 6 legally separated
2	Application mode	Nominal	Code	18 codes, 1–57 (Table A1)	Admission route, e.g. 1st-phase general contingent, over-23 entry, transfer
3	Application order	Ordinal	Rank	0–9 (observed 0–6 and 9)	Preference rank of the course in the student's application; 1 = first choice
4	Course	Nominal	Code	17 codes, 33–9991 (Table A2)	Degree programme enrolled in
5	Daytime/evening attendance	Binary	Flag	1 daytime, 0 evening	Attendance regime
6	Previous qualification	Nominal	Code	17 codes, 1–43 (Table A3)	Highest qualification before entry
7	Previous qualification (grade)	Continuous	Points, 0–200	Observed 95–190	Grade of the previous qualification
8	Nacionality	Nominal	Code	21 codes, 1–109 (Table A4)	Nationality (column name spelled as in the source)
9	Mother's qualification	Nominal	Code	1–44; 29 observed (Table A5)	Mother's highest education level
10	Father's qualification	Nominal	Code	1–44; 34 observed (Table A5)	Father's highest education level
11	Mother's occupation	Nominal	Code	0–194; 32 observed (Table A6)	Mother's occupation group
12	Father's occupation	Nominal	Code	0–195; 46 observed (Table A6)	Father's occupation group
13	Admission grade	Continuous	Points, 0–200	Observed 95–190	Score used for admission to the course
14	Displaced	Binary	Flag	1 yes, 0 no	Student lives away from home to study
15	Educational special needs	Binary	Flag	1 yes, 0 no	Has recognised special educational needs
16	Debtor	Binary	Flag	1 yes, 0 no	Owes money to the institution
17	Tuition fees up to date	Binary	Flag	1 yes, 0 no	Tuition payments are current
18	Gender	Binary	Flag	1 male, 0 female	Student's gender
19	Scholarship holder	Binary	Flag	1 yes, 0 no	Receives a scholarship
20	Age at enrollment	Discrete	Years	17–70	Age when first enrolled
21	International	Binary	Flag	1 yes, 0 no	International student
22	Curricular units 1st sem (credited)	Count	Units	0–20	Units credited from prior study, semester 1
23	Curricular units 1st sem (enrolled)	Count	Units	0–26	Units enrolled, semester 1
24	Curricular units 1st sem (evaluations)	Count	Assessments	0–45	Number of evaluations taken, semester 1
25	Curricular units 1st sem (approved)	Count	Units	0–26	Units passed, semester 1
26	Curricular units 1st sem (grade)	Continuous	Points, 0–20	0–18.9	Average grade in semester 1 (0 if nothing graded)
27	Curricular units 1st sem (without evaluations)	Count	Units	0–12	Enrolled units with no evaluation, semester 1
28	Curricular units 2nd sem (credited)	Count	Units	0–19	Units credited from prior study, semester 2
29	Curricular units 2nd sem (enrolled)	Count	Units	0–23	Units enrolled, semester 2
30	Curricular units 2nd sem (evaluations)	Count	Assessments	0–33	Number of evaluations taken, semester 2
31	Curricular units 2nd sem (approved)	Count	Units	0–20	Units passed, semester 2
32	Curricular units 2nd sem (grade)	Continuous	Points, 0–20	0–18.6	Average grade in semester 2 (0 if nothing graded)
33	Curricular units 2nd sem (without evaluations)	Count	Units	0–12	Enrolled units with no evaluation, semester 2
34	Unemployment rate	Continuous	%	7.6–16.2 (10 values)	National unemployment rate in the enrolment year
35	Inflation rate	Continuous	%	−0.8–3.7 (9 values)	National inflation rate in the enrolment year
36	GDP	Continuous	% growth	−4.06–3.51 (10 values)	National GDP growth in the enrolment year
37	target	Nominal (outcome)	Label	Graduate, Dropout, Enrolled	Status at the end of the normal course duration; the project models Dropout vs the rest


Dataset: `data/raw/students_dropout_academic_success.csv` — 4,424 rows × 37 columns.
