(define (problem p0-pacman)
	(:domain pacman)
	(:objects 
		cell1_1 cell1_2 cell1_3 cell1_4 cell1_5 cell1_6 cell1_7 cell1_8 cell1_9 cell1_10 cell1_11 cell1_12 cell1_13 cell1_14 cell2_14 cell3_1 cell3_2 cell3_3 cell3_4 cell3_5 cell3_6 cell3_8 cell3_10 cell3_11 cell3_12 cell3_13 cell3_14 cell4_6 cell4_7 cell4_8 cell4_9 cell4_10 cell5_1 cell5_2 cell5_3 cell5_4 cell5_5 cell5_6 cell5_7 cell5_10 cell5_12 cell5_14 cell6_2 cell6_5 cell6_9 cell6_10 cell6_11 cell6_12 cell6_13 cell6_14 cell7_1 cell7_2 cell7_3 cell7_5 cell7_6 cell7_8 cell7_9 cell7_13 cell8_1 cell8_3 cell8_6 cell8_9 cell8_11 cell8_12 cell8_13 cell8_14 cell9_1 cell9_5 cell9_6 cell9_11 cell9_13 cell10_1 cell10_3 cell10_6 cell10_7 cell10_9 cell10_11 cell10_13 cell10_14 cell11_1 cell11_2 cell11_3 cell11_4 cell11_6 cell11_9 cell11_13 cell12_2 cell12_4 cell12_6 cell12_7 cell12_8 cell12_9 cell12_10 cell12_11 cell12_12 cell12_13 cell12_14 cell13_1 cell13_2 cell13_4 cell13_5 cell13_6 cell13_7 cell13_8 cell13_12 cell13_14 cell14_1 cell14_5 cell14_7 cell14_8 cell14_9 cell14_11 cell14_12 cell14_14 cell15_1 cell15_2 cell15_4 cell15_5 cell15_7 cell15_8 cell15_11 cell15_12 cell15_13 cell15_14 cell16_1 cell16_2 cell16_3 cell16_4 cell16_7 cell16_8 cell16_10 cell16_11 cell16_13 cell16_14 cell17_1 cell17_3 cell17_4 cell17_6 cell17_7 cell17_8 cell17_10 cell17_14 cell18_1 cell18_3 cell18_7 cell18_8 cell18_9 cell18_10 cell18_11 cell18_13 cell18_14 cell19_1 cell19_2 cell19_3 cell19_4 cell19_5 cell19_6 cell19_7 cell19_8 cell19_9 cell19_11 cell19_13 cell20_2 cell20_6 cell20_9 cell20_11 cell20_12 cell20_13 cell20_14 cell21_1 cell21_2 cell21_4 cell21_6 cell21_8 cell21_9 cell21_12 cell21_14 cell22_2 cell22_4 cell22_9 cell22_10 cell22_14 cell23_1 cell23_2 cell23_3 cell23_4 cell23_6 cell23_9 cell23_12 cell23_14 cell24_2 cell24_6 cell24_7 cell24_9 cell24_10 cell24_12 cell24_13 cell24_14 cell25_1 cell25_2 cell25_3 cell25_4 cell25_5 cell25_6 cell25_10 cell25_13 cell26_1 cell26_3 cell26_5 cell26_8 cell26_9 cell26_10 cell26_11 cell26_12 cell26_13 cell26_14 cell27_5 cell27_6 cell27_7 cell27_8 cell27_9 cell28_1 cell28_2 cell28_3 cell28_4 cell28_5 cell28_7 cell28_9 cell28_10 cell28_11 cell28_12 cell28_13 cell28_14 cell29_1 cell30_1 cell30_2 cell30_3 cell30_4 cell30_5 cell30_6 cell30_7 cell30_8 cell30_9 cell30_10 cell30_11 cell30_12 cell30_13 cell30_14 - cells
		food1 food2 food3 food4 food5 food6 food7 food8 food9 food10 food11 food12 food13 food14 food15 food16 food17 food18 food19 food20 - foods
	)
	(:init 
		(at-pacman cell1_2)
		(at-food food1 cell17_6)
		(at-food food2 cell21_1)
		(at-food food3 cell21_4)
		(at-food food4 cell21_6)
		(at-food food5 cell21_8)
		(at-food food6 cell21_12)
		(at-food food7 cell22_4)
		(at-food food8 cell22_10)
		(at-food food9 cell23_1)
		(at-food food10 cell23_6)
		(at-food food11 cell23_12)
		(at-food food12 cell24_6)
		(at-food food13 cell24_7)
		(at-food food14 cell24_12)
		(at-food food15 cell26_1)
		(at-food food16 cell26_3)
		(at-food food17 cell26_14)
		(at-food food18 cell28_7)
		(at-food food19 cell28_13)
		(at-food food20 cell28_14)
		(has-ghost cell30_13)
		(has-ghost cell30_14)
		(has-capsule cell25_10)
		(connected cell1_1 cell1_2)
		(connected cell1_2 cell1_3)
		(connected cell1_2 cell1_1)
		(connected cell1_3 cell1_4)
		(connected cell1_3 cell1_2)
		(connected cell1_4 cell1_5)
		(connected cell1_4 cell1_3)
		(connected cell1_5 cell1_6)
		(connected cell1_5 cell1_4)
		(connected cell1_6 cell1_7)
		(connected cell1_6 cell1_5)
		(connected cell1_7 cell1_8)
		(connected cell1_7 cell1_6)
		(connected cell1_8 cell1_9)
		(connected cell1_8 cell1_7)
		(connected cell1_9 cell1_10)
		(connected cell1_9 cell1_8)
		(connected cell1_10 cell1_11)
		(connected cell1_10 cell1_9)
		(connected cell1_11 cell1_12)
		(connected cell1_11 cell1_10)
		(connected cell1_12 cell1_13)
		(connected cell1_12 cell1_11)
		(connected cell1_13 cell1_14)
		(connected cell1_13 cell1_12)
		(connected cell1_14 cell2_14)
		(connected cell1_14 cell1_13)
		(connected cell2_14 cell3_14)
		(connected cell2_14 cell1_14)
		(connected cell3_1 cell3_2)
		(connected cell3_2 cell3_3)
		(connected cell3_2 cell3_1)
		(connected cell3_3 cell3_4)
		(connected cell3_3 cell3_2)
		(connected cell3_4 cell3_5)
		(connected cell3_4 cell3_3)
		(connected cell3_5 cell3_6)
		(connected cell3_5 cell3_4)
		(connected cell3_6 cell4_6)
		(connected cell3_6 cell3_5)
		(connected cell3_8 cell4_8)
		(connected cell3_10 cell4_10)
		(connected cell3_10 cell3_11)
		(connected cell3_11 cell3_12)
		(connected cell3_11 cell3_10)
		(connected cell3_12 cell3_13)
		(connected cell3_12 cell3_11)
		(connected cell3_13 cell3_14)
		(connected cell3_13 cell3_12)
		(connected cell3_14 cell2_14)
		(connected cell3_14 cell3_13)
		(connected cell4_6 cell5_6)
		(connected cell4_6 cell3_6)
		(connected cell4_6 cell4_7)
		(connected cell4_7 cell5_7)
		(connected cell4_7 cell4_8)
		(connected cell4_7 cell4_6)
		(connected cell4_8 cell3_8)
		(connected cell4_8 cell4_9)
		(connected cell4_8 cell4_7)
		(connected cell4_9 cell4_10)
		(connected cell4_9 cell4_8)
		(connected cell4_10 cell5_10)
		(connected cell4_10 cell3_10)
		(connected cell4_10 cell4_9)
		(connected cell5_1 cell5_2)
		(connected cell5_2 cell6_2)
		(connected cell5_2 cell5_3)
		(connected cell5_2 cell5_1)
		(connected cell5_3 cell5_4)
		(connected cell5_3 cell5_2)
		(connected cell5_4 cell5_5)
		(connected cell5_4 cell5_3)
		(connected cell5_5 cell6_5)
		(connected cell5_5 cell5_6)
		(connected cell5_5 cell5_4)
		(connected cell5_6 cell4_6)
		(connected cell5_6 cell5_7)
		(connected cell5_6 cell5_5)
		(connected cell5_7 cell4_7)
		(connected cell5_7 cell5_6)
		(connected cell5_10 cell6_10)
		(connected cell5_10 cell4_10)
		(connected cell5_12 cell6_12)
		(connected cell5_14 cell6_14)
		(connected cell6_2 cell7_2)
		(connected cell6_2 cell5_2)
		(connected cell6_5 cell7_5)
		(connected cell6_5 cell5_5)
		(connected cell6_9 cell7_9)
		(connected cell6_9 cell6_10)
		(connected cell6_10 cell5_10)
		(connected cell6_10 cell6_11)
		(connected cell6_10 cell6_9)
		(connected cell6_11 cell6_12)
		(connected cell6_11 cell6_10)
		(connected cell6_12 cell5_12)
		(connected cell6_12 cell6_13)
		(connected cell6_12 cell6_11)
		(connected cell6_13 cell7_13)
		(connected cell6_13 cell6_14)
		(connected cell6_13 cell6_12)
		(connected cell6_14 cell5_14)
		(connected cell6_14 cell6_13)
		(connected cell7_1 cell8_1)
		(connected cell7_1 cell7_2)
		(connected cell7_2 cell6_2)
		(connected cell7_2 cell7_3)
		(connected cell7_2 cell7_1)
		(connected cell7_3 cell8_3)
		(connected cell7_3 cell7_2)
		(connected cell7_5 cell6_5)
		(connected cell7_5 cell7_6)
		(connected cell7_6 cell8_6)
		(connected cell7_6 cell7_5)
		(connected cell7_8 cell7_9)
		(connected cell7_9 cell8_9)
		(connected cell7_9 cell6_9)
		(connected cell7_9 cell7_8)
		(connected cell7_13 cell8_13)
		(connected cell7_13 cell6_13)
		(connected cell8_1 cell9_1)
		(connected cell8_1 cell7_1)
		(connected cell8_3 cell7_3)
		(connected cell8_6 cell9_6)
		(connected cell8_6 cell7_6)
		(connected cell8_9 cell7_9)
		(connected cell8_11 cell9_11)
		(connected cell8_11 cell8_12)
		(connected cell8_12 cell8_13)
		(connected cell8_12 cell8_11)
		(connected cell8_13 cell9_13)
		(connected cell8_13 cell7_13)
		(connected cell8_13 cell8_14)
		(connected cell8_13 cell8_12)
		(connected cell8_14 cell8_13)
		(connected cell9_1 cell10_1)
		(connected cell9_1 cell8_1)
		(connected cell9_5 cell9_6)
		(connected cell9_6 cell10_6)
		(connected cell9_6 cell8_6)
		(connected cell9_6 cell9_5)
		(connected cell9_11 cell10_11)
		(connected cell9_11 cell8_11)
		(connected cell9_13 cell10_13)
		(connected cell9_13 cell8_13)
		(connected cell10_1 cell11_1)
		(connected cell10_1 cell9_1)
		(connected cell10_3 cell11_3)
		(connected cell10_6 cell11_6)
		(connected cell10_6 cell9_6)
		(connected cell10_6 cell10_7)
		(connected cell10_7 cell10_6)
		(connected cell10_9 cell11_9)
		(connected cell10_11 cell9_11)
		(connected cell10_13 cell11_13)
		(connected cell10_13 cell9_13)
		(connected cell10_13 cell10_14)
		(connected cell10_14 cell10_13)
		(connected cell11_1 cell10_1)
		(connected cell11_1 cell11_2)
		(connected cell11_2 cell12_2)
		(connected cell11_2 cell11_3)
		(connected cell11_2 cell11_1)
		(connected cell11_3 cell10_3)
		(connected cell11_3 cell11_4)
		(connected cell11_3 cell11_2)
		(connected cell11_4 cell12_4)
		(connected cell11_4 cell11_3)
		(connected cell11_6 cell12_6)
		(connected cell11_6 cell10_6)
		(connected cell11_9 cell12_9)
		(connected cell11_9 cell10_9)
		(connected cell11_13 cell12_13)
		(connected cell11_13 cell10_13)
		(connected cell12_2 cell13_2)
		(connected cell12_2 cell11_2)
		(connected cell12_4 cell13_4)
		(connected cell12_4 cell11_4)
		(connected cell12_6 cell13_6)
		(connected cell12_6 cell11_6)
		(connected cell12_6 cell12_7)
		(connected cell12_7 cell13_7)
		(connected cell12_7 cell12_8)
		(connected cell12_7 cell12_6)
		(connected cell12_8 cell13_8)
		(connected cell12_8 cell12_9)
		(connected cell12_8 cell12_7)
		(connected cell12_9 cell11_9)
		(connected cell12_9 cell12_10)
		(connected cell12_9 cell12_8)
		(connected cell12_10 cell12_11)
		(connected cell12_10 cell12_9)
		(connected cell12_11 cell12_12)
		(connected cell12_11 cell12_10)
		(connected cell12_12 cell13_12)
		(connected cell12_12 cell12_13)
		(connected cell12_12 cell12_11)
		(connected cell12_13 cell11_13)
		(connected cell12_13 cell12_14)
		(connected cell12_13 cell12_12)
		(connected cell12_14 cell13_14)
		(connected cell12_14 cell12_13)
		(connected cell13_1 cell14_1)
		(connected cell13_1 cell13_2)
		(connected cell13_2 cell12_2)
		(connected cell13_2 cell13_1)
		(connected cell13_4 cell12_4)
		(connected cell13_4 cell13_5)
		(connected cell13_5 cell14_5)
		(connected cell13_5 cell13_6)
		(connected cell13_5 cell13_4)
		(connected cell13_6 cell12_6)
		(connected cell13_6 cell13_7)
		(connected cell13_6 cell13_5)
		(connected cell13_7 cell14_7)
		(connected cell13_7 cell12_7)
		(connected cell13_7 cell13_8)
		(connected cell13_7 cell13_6)
		(connected cell13_8 cell14_8)
		(connected cell13_8 cell12_8)
		(connected cell13_8 cell13_7)
		(connected cell13_12 cell14_12)
		(connected cell13_12 cell12_12)
		(connected cell13_14 cell14_14)
		(connected cell13_14 cell12_14)
		(connected cell14_1 cell15_1)
		(connected cell14_1 cell13_1)
		(connected cell14_5 cell15_5)
		(connected cell14_5 cell13_5)
		(connected cell14_7 cell15_7)
		(connected cell14_7 cell13_7)
		(connected cell14_7 cell14_8)
		(connected cell14_8 cell15_8)
		(connected cell14_8 cell13_8)
		(connected cell14_8 cell14_9)
		(connected cell14_8 cell14_7)
		(connected cell14_9 cell14_8)
		(connected cell14_11 cell15_11)
		(connected cell14_11 cell14_12)
		(connected cell14_12 cell15_12)
		(connected cell14_12 cell13_12)
		(connected cell14_12 cell14_11)
		(connected cell14_14 cell15_14)
		(connected cell14_14 cell13_14)
		(connected cell15_1 cell16_1)
		(connected cell15_1 cell14_1)
		(connected cell15_1 cell15_2)
		(connected cell15_2 cell16_2)
		(connected cell15_2 cell15_1)
		(connected cell15_4 cell16_4)
		(connected cell15_4 cell15_5)
		(connected cell15_5 cell14_5)
		(connected cell15_5 cell15_4)
		(connected cell15_7 cell16_7)
		(connected cell15_7 cell14_7)
		(connected cell15_7 cell15_8)
		(connected cell15_8 cell16_8)
		(connected cell15_8 cell14_8)
		(connected cell15_8 cell15_7)
		(connected cell15_11 cell16_11)
		(connected cell15_11 cell14_11)
		(connected cell15_11 cell15_12)
		(connected cell15_12 cell14_12)
		(connected cell15_12 cell15_13)
		(connected cell15_12 cell15_11)
		(connected cell15_13 cell16_13)
		(connected cell15_13 cell15_14)
		(connected cell15_13 cell15_12)
		(connected cell15_14 cell16_14)
		(connected cell15_14 cell14_14)
		(connected cell15_14 cell15_13)
		(connected cell16_1 cell17_1)
		(connected cell16_1 cell15_1)
		(connected cell16_1 cell16_2)
		(connected cell16_2 cell15_2)
		(connected cell16_2 cell16_3)
		(connected cell16_2 cell16_1)
		(connected cell16_3 cell17_3)
		(connected cell16_3 cell16_4)
		(connected cell16_3 cell16_2)
		(connected cell16_4 cell17_4)
		(connected cell16_4 cell15_4)
		(connected cell16_4 cell16_3)
		(connected cell16_7 cell17_7)
		(connected cell16_7 cell15_7)
		(connected cell16_7 cell16_8)
		(connected cell16_8 cell17_8)
		(connected cell16_8 cell15_8)
		(connected cell16_8 cell16_7)
		(connected cell16_10 cell17_10)
		(connected cell16_10 cell16_11)
		(connected cell16_11 cell15_11)
		(connected cell16_11 cell16_10)
		(connected cell16_13 cell15_13)
		(connected cell16_13 cell16_14)
		(connected cell16_14 cell17_14)
		(connected cell16_14 cell15_14)
		(connected cell16_14 cell16_13)
		(connected cell17_1 cell18_1)
		(connected cell17_1 cell16_1)
		(connected cell17_3 cell18_3)
		(connected cell17_3 cell16_3)
		(connected cell17_3 cell17_4)
		(connected cell17_4 cell16_4)
		(connected cell17_4 cell17_3)
		(connected cell17_6 cell17_7)
		(connected cell17_7 cell18_7)
		(connected cell17_7 cell16_7)
		(connected cell17_7 cell17_8)
		(connected cell17_7 cell17_6)
		(connected cell17_8 cell18_8)
		(connected cell17_8 cell16_8)
		(connected cell17_8 cell17_7)
		(connected cell17_10 cell18_10)
		(connected cell17_10 cell16_10)
		(connected cell17_14 cell18_14)
		(connected cell17_14 cell16_14)
		(connected cell18_1 cell19_1)
		(connected cell18_1 cell17_1)
		(connected cell18_3 cell19_3)
		(connected cell18_3 cell17_3)
		(connected cell18_7 cell19_7)
		(connected cell18_7 cell17_7)
		(connected cell18_7 cell18_8)
		(connected cell18_8 cell19_8)
		(connected cell18_8 cell17_8)
		(connected cell18_8 cell18_9)
		(connected cell18_8 cell18_7)
		(connected cell18_9 cell19_9)
		(connected cell18_9 cell18_10)
		(connected cell18_9 cell18_8)
		(connected cell18_10 cell17_10)
		(connected cell18_10 cell18_11)
		(connected cell18_10 cell18_9)
		(connected cell18_11 cell19_11)
		(connected cell18_11 cell18_10)
		(connected cell18_13 cell19_13)
		(connected cell18_13 cell18_14)
		(connected cell18_14 cell17_14)
		(connected cell18_14 cell18_13)
		(connected cell19_1 cell18_1)
		(connected cell19_1 cell19_2)
		(connected cell19_2 cell20_2)
		(connected cell19_2 cell19_3)
		(connected cell19_2 cell19_1)
		(connected cell19_3 cell18_3)
		(connected cell19_3 cell19_4)
		(connected cell19_3 cell19_2)
		(connected cell19_4 cell19_5)
		(connected cell19_4 cell19_3)
		(connected cell19_5 cell19_6)
		(connected cell19_5 cell19_4)
		(connected cell19_6 cell20_6)
		(connected cell19_6 cell19_7)
		(connected cell19_6 cell19_5)
		(connected cell19_7 cell18_7)
		(connected cell19_7 cell19_8)
		(connected cell19_7 cell19_6)
		(connected cell19_8 cell18_8)
		(connected cell19_8 cell19_9)
		(connected cell19_8 cell19_7)
		(connected cell19_9 cell20_9)
		(connected cell19_9 cell18_9)
		(connected cell19_9 cell19_8)
		(connected cell19_11 cell20_11)
		(connected cell19_11 cell18_11)
		(connected cell19_13 cell20_13)
		(connected cell19_13 cell18_13)
		(connected cell20_2 cell21_2)
		(connected cell20_2 cell19_2)
		(connected cell20_6 cell21_6)
		(connected cell20_6 cell19_6)
		(connected cell20_9 cell21_9)
		(connected cell20_9 cell19_9)
		(connected cell20_11 cell19_11)
		(connected cell20_11 cell20_12)
		(connected cell20_12 cell21_12)
		(connected cell20_12 cell20_13)
		(connected cell20_12 cell20_11)
		(connected cell20_13 cell19_13)
		(connected cell20_13 cell20_14)
		(connected cell20_13 cell20_12)
		(connected cell20_14 cell21_14)
		(connected cell20_14 cell20_13)
		(connected cell21_1 cell21_2)
		(connected cell21_2 cell22_2)
		(connected cell21_2 cell20_2)
		(connected cell21_2 cell21_1)
		(connected cell21_4 cell22_4)
		(connected cell21_6 cell20_6)
		(connected cell21_8 cell21_9)
		(connected cell21_9 cell22_9)
		(connected cell21_9 cell20_9)
		(connected cell21_9 cell21_8)
		(connected cell21_12 cell20_12)
		(connected cell21_14 cell22_14)
		(connected cell21_14 cell20_14)
		(connected cell22_2 cell23_2)
		(connected cell22_2 cell21_2)
		(connected cell22_4 cell23_4)
		(connected cell22_4 cell21_4)
		(connected cell22_9 cell23_9)
		(connected cell22_9 cell21_9)
		(connected cell22_9 cell22_10)
		(connected cell22_10 cell22_9)
		(connected cell22_14 cell23_14)
		(connected cell22_14 cell21_14)
		(connected cell23_1 cell23_2)
		(connected cell23_2 cell24_2)
		(connected cell23_2 cell22_2)
		(connected cell23_2 cell23_3)
		(connected cell23_2 cell23_1)
		(connected cell23_3 cell23_4)
		(connected cell23_3 cell23_2)
		(connected cell23_4 cell22_4)
		(connected cell23_4 cell23_3)
		(connected cell23_6 cell24_6)
		(connected cell23_9 cell24_9)
		(connected cell23_9 cell22_9)
		(connected cell23_12 cell24_12)
		(connected cell23_14 cell24_14)
		(connected cell23_14 cell22_14)
		(connected cell24_2 cell25_2)
		(connected cell24_2 cell23_2)
		(connected cell24_6 cell25_6)
		(connected cell24_6 cell23_6)
		(connected cell24_6 cell24_7)
		(connected cell24_7 cell24_6)
		(connected cell24_9 cell23_9)
		(connected cell24_9 cell24_10)
		(connected cell24_10 cell25_10)
		(connected cell24_10 cell24_9)
		(connected cell24_12 cell23_12)
		(connected cell24_12 cell24_13)
		(connected cell24_13 cell25_13)
		(connected cell24_13 cell24_14)
		(connected cell24_13 cell24_12)
		(connected cell24_14 cell23_14)
		(connected cell24_14 cell24_13)
		(connected cell25_1 cell26_1)
		(connected cell25_1 cell25_2)
		(connected cell25_2 cell24_2)
		(connected cell25_2 cell25_3)
		(connected cell25_2 cell25_1)
		(connected cell25_3 cell26_3)
		(connected cell25_3 cell25_4)
		(connected cell25_3 cell25_2)
		(connected cell25_4 cell25_5)
		(connected cell25_4 cell25_3)
		(connected cell25_5 cell26_5)
		(connected cell25_5 cell25_6)
		(connected cell25_5 cell25_4)
		(connected cell25_6 cell24_6)
		(connected cell25_6 cell25_5)
		(connected cell25_10 cell26_10)
		(connected cell25_10 cell24_10)
		(connected cell25_13 cell26_13)
		(connected cell25_13 cell24_13)
		(connected cell26_1 cell25_1)
		(connected cell26_3 cell25_3)
		(connected cell26_5 cell27_5)
		(connected cell26_5 cell25_5)
		(connected cell26_8 cell27_8)
		(connected cell26_8 cell26_9)
		(connected cell26_9 cell27_9)
		(connected cell26_9 cell26_10)
		(connected cell26_9 cell26_8)
		(connected cell26_10 cell25_10)
		(connected cell26_10 cell26_11)
		(connected cell26_10 cell26_9)
		(connected cell26_11 cell26_12)
		(connected cell26_11 cell26_10)
		(connected cell26_12 cell26_13)
		(connected cell26_12 cell26_11)
		(connected cell26_13 cell25_13)
		(connected cell26_13 cell26_14)
		(connected cell26_13 cell26_12)
		(connected cell26_14 cell26_13)
		(connected cell27_5 cell28_5)
		(connected cell27_5 cell26_5)
		(connected cell27_5 cell27_6)
		(connected cell27_6 cell27_7)
		(connected cell27_6 cell27_5)
		(connected cell27_7 cell28_7)
		(connected cell27_7 cell27_8)
		(connected cell27_7 cell27_6)
		(connected cell27_8 cell26_8)
		(connected cell27_8 cell27_9)
		(connected cell27_8 cell27_7)
		(connected cell27_9 cell28_9)
		(connected cell27_9 cell26_9)
		(connected cell27_9 cell27_8)
		(connected cell28_1 cell29_1)
		(connected cell28_1 cell28_2)
		(connected cell28_2 cell28_3)
		(connected cell28_2 cell28_1)
		(connected cell28_3 cell28_4)
		(connected cell28_3 cell28_2)
		(connected cell28_4 cell28_5)
		(connected cell28_4 cell28_3)
		(connected cell28_5 cell27_5)
		(connected cell28_5 cell28_4)
		(connected cell28_7 cell27_7)
		(connected cell28_9 cell27_9)
		(connected cell28_9 cell28_10)
		(connected cell28_10 cell28_11)
		(connected cell28_10 cell28_9)
		(connected cell28_11 cell28_12)
		(connected cell28_11 cell28_10)
		(connected cell28_12 cell28_13)
		(connected cell28_12 cell28_11)
		(connected cell28_13 cell28_14)
		(connected cell28_13 cell28_12)
		(connected cell28_14 cell28_13)
		(connected cell29_1 cell30_1)
		(connected cell29_1 cell28_1)
		(connected cell30_1 cell29_1)
		(connected cell30_1 cell30_2)
		(connected cell30_2 cell30_3)
		(connected cell30_2 cell30_1)
		(connected cell30_3 cell30_4)
		(connected cell30_3 cell30_2)
		(connected cell30_4 cell30_5)
		(connected cell30_4 cell30_3)
		(connected cell30_5 cell30_6)
		(connected cell30_5 cell30_4)
		(connected cell30_6 cell30_7)
		(connected cell30_6 cell30_5)
		(connected cell30_7 cell30_8)
		(connected cell30_7 cell30_6)
		(connected cell30_8 cell30_9)
		(connected cell30_8 cell30_7)
		(connected cell30_9 cell30_10)
		(connected cell30_9 cell30_8)
		(connected cell30_10 cell30_11)
		(connected cell30_10 cell30_9)
		(connected cell30_11 cell30_12)
		(connected cell30_11 cell30_10)
		(connected cell30_12 cell30_13)
		(connected cell30_12 cell30_11)
		(connected cell30_13 cell30_14)
		(connected cell30_13 cell30_12)
		(connected cell30_14 cell30_13)
	)
	(:goal (and
		(carrying-food)
	))
)