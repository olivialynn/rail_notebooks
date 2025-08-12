GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

A quick demo of running GPz on the typical test data. You should have
installed rail_gpz_v1 (we highly recommend that you do this from within
a custom conda environment so that all dependencies for package versions
are met), either by cloning and installing from github, or with:

::

   pip install pz-rail-gpz-v1

As RAIL is a namespace package, installing rail_gpz_v1 will make
``GPzInformer`` and ``GPzEstimator`` available, and they can be imported
via:

::

   from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

Let’s start with all of our necessary imports:

.. code:: ipython3

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import rail
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # set up the DataStore to keep track of data
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = DS.read_file("training_data", TableHandle, trainFile)
    test_data = DS.read_file("test_data", TableHandle, testFile)

Now, we need to set up the stage that will run GPz. We begin by defining
a dictionary with the config options for the algorithm. There are
sensible defaults set, we will override several of these as an example
of how to do this. Config parameters not set in the dictionary will
automatically be set to their default values.

.. code:: ipython3

    gpz_train_dict = dict(n_basis=60, trainfrac=0.8, csl_method="normal", max_iter=150, hdf5_groupname="photometry") 

Let’s set up the training stage. We need to provide a name for the stage
for ceci, as well as a name for the model file that will be written by
the stage. We also include the arguments in the dictionary we wrote
above as additional arguments:

.. code:: ipython3

    # set up the stage to run our GPZ_training
    pz_train = GPzInformer.make_stage(name="GPz_Train", model="GPz_model.pkl", **gpz_train_dict)

We are now ready to run the stage to create the model. We will use the
training data from ``test_dc2_training_9816.hdf5``, which contains
10,225 galaxies drawn from healpix 9816 from the cosmoDC2_v1.1.4
dataset, to train the model. Note that we read this data in called
``train_data`` in the DataStore. Note that we set ``trainfrac`` to 0.8,
so 80% of the data will be used in the “main” training, but 20% will be
reserved by ``GPzInformer`` to determine a SIGMA parameter. We set
``max_iter`` to 150, so we will see 150 steps where the stage tries to
maximize the likelihood. We run the stage as follows:

.. code:: ipython3

    %%time
    pz_train.inform(training_data)


.. parsed-literal::

    Inserting handle into data store.  input: None, GPz_Train
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4448499e-01	 3.2118205e-01	-3.3473965e-01	 3.1740065e-01	[-3.2813947e-01]	 4.6330833e-01


.. parsed-literal::

       2	-2.7380897e-01	 3.1006761e-01	-2.4936514e-01	 3.0859955e-01	[-2.4530300e-01]	 2.3318076e-01


.. parsed-literal::

       3	-2.3519269e-01	 2.9245760e-01	-1.9554683e-01	 2.9147922e-01	[-1.9303465e-01]	 2.8579974e-01


.. parsed-literal::

       4	-1.8494808e-01	 2.6802788e-01	-1.4088174e-01	 2.6336616e-01	[-1.1833995e-01]	 2.0805407e-01


.. parsed-literal::

       5	-1.1505585e-01	 2.5736536e-01	-8.4066285e-02	 2.5312612e-01	[-6.2368291e-02]	 2.1954799e-01


.. parsed-literal::

       6	-7.1137908e-02	 2.5190907e-01	-4.2167408e-02	 2.4686530e-01	[-2.4051304e-02]	 2.2088695e-01


.. parsed-literal::

       7	-5.5254425e-02	 2.4931779e-01	-3.0407439e-02	 2.4497951e-01	[-1.3297754e-02]	 2.1487093e-01
       8	-3.9918803e-02	 2.4668397e-01	-1.9544942e-02	 2.4342100e-01	[-5.0311590e-03]	 1.9555569e-01


.. parsed-literal::

       9	-2.8797349e-02	 2.4454719e-01	-1.0411884e-02	 2.4289842e-01	[-1.6594421e-03]	 1.9271755e-01


.. parsed-literal::

      10	-1.9437562e-02	 2.4292620e-01	-3.9446475e-03	 2.4152960e-01	[ 5.6292201e-03]	 2.1623898e-01


.. parsed-literal::

      11	-1.4622965e-02	 2.4217146e-01	 2.9776590e-04	 2.4056907e-01	[ 1.0037455e-02]	 2.1849918e-01


.. parsed-literal::

      12	-1.0510476e-02	 2.4155076e-01	 3.5399223e-03	 2.3936848e-01	[ 1.4880471e-02]	 2.0701075e-01


.. parsed-literal::

      13	-6.6092040e-03	 2.4091930e-01	 7.1769771e-03	 2.3836050e-01	[ 1.9192428e-02]	 2.1076369e-01


.. parsed-literal::

      14	 1.0914486e-01	 2.2588290e-01	 1.3041136e-01	 2.2382301e-01	[ 1.3711927e-01]	 4.4203019e-01


.. parsed-literal::

      15	 1.7735009e-01	 2.2139107e-01	 1.9956768e-01	 2.2120693e-01	[ 2.0397149e-01]	 2.1134329e-01


.. parsed-literal::

      16	 2.3379269e-01	 2.1660192e-01	 2.6261452e-01	 2.1658623e-01	[ 2.6495940e-01]	 2.1241760e-01
      17	 2.9918373e-01	 2.1547887e-01	 3.2951525e-01	 2.1468859e-01	[ 3.3746934e-01]	 1.8860102e-01


.. parsed-literal::

      18	 3.3438794e-01	 2.1397169e-01	 3.6535500e-01	 2.1208400e-01	[ 3.7745282e-01]	 2.2038341e-01


.. parsed-literal::

      19	 3.7967044e-01	 2.1161047e-01	 4.1182817e-01	 2.0767133e-01	[ 4.2388667e-01]	 2.1880364e-01


.. parsed-literal::

      20	 4.5345794e-01	 2.0952404e-01	 4.8748923e-01	 2.0639409e-01	[ 4.9262497e-01]	 2.1390557e-01


.. parsed-literal::

      21	 5.5540851e-01	 2.0658864e-01	 5.9263969e-01	 2.0501509e-01	[ 5.7013276e-01]	 2.1107745e-01


.. parsed-literal::

      22	 6.1314218e-01	 2.0349788e-01	 6.5298684e-01	 1.9984800e-01	[ 6.2234257e-01]	 2.0488715e-01


.. parsed-literal::

      23	 6.5463875e-01	 1.9783821e-01	 6.9349548e-01	 1.9551987e-01	[ 6.6151974e-01]	 2.0445204e-01


.. parsed-literal::

      24	 7.0242979e-01	 1.9129291e-01	 7.4133047e-01	 1.8918326e-01	[ 7.0509462e-01]	 2.1321130e-01


.. parsed-literal::

      25	 7.4038794e-01	 1.9973096e-01	 7.7768819e-01	 1.9364758e-01	[ 7.6071182e-01]	 2.0100284e-01


.. parsed-literal::

      26	 7.8035698e-01	 1.9587598e-01	 8.1809476e-01	 1.9153348e-01	[ 7.9468901e-01]	 2.1771884e-01


.. parsed-literal::

      27	 8.0599556e-01	 1.9127213e-01	 8.4562654e-01	 1.8718172e-01	[ 8.1291065e-01]	 2.1167898e-01


.. parsed-literal::

      28	 8.4495999e-01	 1.8753562e-01	 8.8525951e-01	 1.8353686e-01	[ 8.5124231e-01]	 2.0748186e-01


.. parsed-literal::

      29	 8.6896116e-01	 1.8540408e-01	 9.0934032e-01	 1.8231074e-01	[ 8.8817622e-01]	 2.1500349e-01


.. parsed-literal::

      30	 8.8723868e-01	 1.8367625e-01	 9.2725687e-01	 1.8053661e-01	[ 9.0633249e-01]	 2.0389700e-01


.. parsed-literal::

      31	 8.9828036e-01	 1.8243292e-01	 9.3840279e-01	 1.7946475e-01	[ 9.1405708e-01]	 2.1280622e-01


.. parsed-literal::

      32	 9.2275741e-01	 1.7968539e-01	 9.6385233e-01	 1.7644804e-01	[ 9.2704775e-01]	 2.0900130e-01
      33	 9.3864080e-01	 1.7981368e-01	 9.8019887e-01	 1.7681099e-01	[ 9.3381288e-01]	 1.8359447e-01


.. parsed-literal::

      34	 9.5580091e-01	 1.7616744e-01	 9.9734462e-01	 1.7308969e-01	[ 9.5270550e-01]	 2.0126224e-01
      35	 9.6834860e-01	 1.7501557e-01	 1.0098129e+00	 1.7111818e-01	[ 9.7120034e-01]	 1.8945146e-01


.. parsed-literal::

      36	 9.8158725e-01	 1.7319028e-01	 1.0234507e+00	 1.6932447e-01	[ 9.8269334e-01]	 2.0480347e-01
      37	 1.0019994e+00	 1.7118223e-01	 1.0446775e+00	 1.6741720e-01	[ 9.8754929e-01]	 1.7902303e-01


.. parsed-literal::

      38	 1.0199089e+00	 1.6974845e-01	 1.0634869e+00	 1.6621178e-01	[ 9.9735931e-01]	 2.0996308e-01


.. parsed-literal::

      39	 1.0331387e+00	 1.6807990e-01	 1.0774211e+00	 1.6459829e-01	[ 1.0001536e+00]	 2.0642734e-01
      40	 1.0426395e+00	 1.6636316e-01	 1.0869721e+00	 1.6310391e-01	[ 1.0080106e+00]	 1.7386651e-01


.. parsed-literal::

      41	 1.0562301e+00	 1.6341446e-01	 1.1009090e+00	 1.6046986e-01	[ 1.0188874e+00]	 1.8525457e-01


.. parsed-literal::

      42	 1.0686026e+00	 1.6161528e-01	 1.1136625e+00	 1.5887202e-01	[ 1.0301984e+00]	 2.1395063e-01


.. parsed-literal::

      43	 1.0826391e+00	 1.6037647e-01	 1.1287222e+00	 1.5731971e-01	[ 1.0502837e+00]	 2.1529937e-01


.. parsed-literal::

      44	 1.0957622e+00	 1.5886057e-01	 1.1418189e+00	 1.5579603e-01	[ 1.0614819e+00]	 2.0893288e-01
      45	 1.1038206e+00	 1.5796566e-01	 1.1497406e+00	 1.5501740e-01	[ 1.0700997e+00]	 1.7803359e-01


.. parsed-literal::

      46	 1.1178347e+00	 1.5666567e-01	 1.1636984e+00	 1.5354709e-01	[ 1.0855855e+00]	 2.0631814e-01
      47	 1.1290451e+00	 1.5464598e-01	 1.1755590e+00	 1.5171847e-01	[ 1.1042187e+00]	 1.7287135e-01


.. parsed-literal::

      48	 1.1419018e+00	 1.5415591e-01	 1.1882701e+00	 1.5132329e-01	[ 1.1223147e+00]	 1.9953918e-01
      49	 1.1533896e+00	 1.5300639e-01	 1.2001070e+00	 1.5043775e-01	[ 1.1381216e+00]	 1.7806935e-01


.. parsed-literal::

      50	 1.1633237e+00	 1.5168680e-01	 1.2104901e+00	 1.4935620e-01	[ 1.1460455e+00]	 2.0879006e-01


.. parsed-literal::

      51	 1.1738479e+00	 1.5000886e-01	 1.2213190e+00	 1.4839435e-01	[ 1.1605822e+00]	 2.1569657e-01
      52	 1.1827438e+00	 1.4902744e-01	 1.2303810e+00	 1.4832829e-01	[ 1.1713059e+00]	 2.0381403e-01


.. parsed-literal::

      53	 1.1920753e+00	 1.4776537e-01	 1.2396187e+00	 1.4719711e-01	[ 1.1810662e+00]	 1.8561840e-01


.. parsed-literal::

      54	 1.2013986e+00	 1.4670757e-01	 1.2490765e+00	 1.4669261e-01	[ 1.1930380e+00]	 2.1295357e-01


.. parsed-literal::

      55	 1.2099909e+00	 1.4540860e-01	 1.2577261e+00	 1.4533214e-01	[ 1.2028114e+00]	 2.1138883e-01
      56	 1.2190099e+00	 1.4456882e-01	 1.2667983e+00	 1.4484259e-01	[ 1.2107757e+00]	 2.0030808e-01


.. parsed-literal::

      57	 1.2304462e+00	 1.4360381e-01	 1.2786013e+00	 1.4451814e-01	[ 1.2196379e+00]	 1.7418981e-01


.. parsed-literal::

      58	 1.2421203e+00	 1.4305897e-01	 1.2904212e+00	 1.4363371e-01	[ 1.2347420e+00]	 2.0998430e-01


.. parsed-literal::

      59	 1.2521643e+00	 1.4280534e-01	 1.3006802e+00	 1.4378963e-01	[ 1.2407411e+00]	 2.0729661e-01


.. parsed-literal::

      60	 1.2635391e+00	 1.4211099e-01	 1.3125103e+00	 1.4304498e-01	[ 1.2429744e+00]	 2.1639609e-01


.. parsed-literal::

      61	 1.2709009e+00	 1.4079491e-01	 1.3200066e+00	 1.4182183e-01	  1.2345003e+00 	 2.1584868e-01


.. parsed-literal::

      62	 1.2781993e+00	 1.4028034e-01	 1.3272358e+00	 1.4152860e-01	  1.2311146e+00 	 2.1446085e-01


.. parsed-literal::

      63	 1.2839993e+00	 1.4020280e-01	 1.3332343e+00	 1.4123935e-01	  1.2244510e+00 	 2.1010113e-01


.. parsed-literal::

      64	 1.2890195e+00	 1.3963093e-01	 1.3384098e+00	 1.4097531e-01	  1.2198923e+00 	 2.0869732e-01


.. parsed-literal::

      65	 1.2961920e+00	 1.3933726e-01	 1.3457530e+00	 1.4071351e-01	  1.2214102e+00 	 2.0412850e-01


.. parsed-literal::

      66	 1.3033649e+00	 1.3896539e-01	 1.3530806e+00	 1.4027481e-01	  1.2259267e+00 	 2.0937300e-01
      67	 1.3085030e+00	 1.3788854e-01	 1.3585189e+00	 1.3991913e-01	  1.2291548e+00 	 1.8275738e-01


.. parsed-literal::

      68	 1.3139329e+00	 1.3743221e-01	 1.3638745e+00	 1.3925991e-01	  1.2376345e+00 	 1.9682813e-01


.. parsed-literal::

      69	 1.3197021e+00	 1.3676120e-01	 1.3698599e+00	 1.3833184e-01	  1.2416639e+00 	 2.1878767e-01
      70	 1.3248809e+00	 1.3604785e-01	 1.3751855e+00	 1.3765258e-01	[ 1.2435703e+00]	 2.0268989e-01


.. parsed-literal::

      71	 1.3326323e+00	 1.3517798e-01	 1.3831478e+00	 1.3710652e-01	  1.2431965e+00 	 1.8544507e-01
      72	 1.3389563e+00	 1.3455559e-01	 1.3895965e+00	 1.3613572e-01	[ 1.2499809e+00]	 1.8988824e-01


.. parsed-literal::

      73	 1.3442830e+00	 1.3446922e-01	 1.3947629e+00	 1.3594125e-01	[ 1.2545427e+00]	 2.0825100e-01
      74	 1.3507942e+00	 1.3412491e-01	 1.4014509e+00	 1.3546122e-01	  1.2540723e+00 	 1.8676686e-01


.. parsed-literal::

      75	 1.3559724e+00	 1.3363460e-01	 1.4067239e+00	 1.3423780e-01	[ 1.2559844e+00]	 1.6875529e-01


.. parsed-literal::

      76	 1.3602848e+00	 1.3327429e-01	 1.4110353e+00	 1.3371533e-01	  1.2556769e+00 	 2.0659971e-01
      77	 1.3650393e+00	 1.3236282e-01	 1.4158230e+00	 1.3275816e-01	[ 1.2562133e+00]	 1.7450762e-01


.. parsed-literal::

      78	 1.3695538e+00	 1.3192094e-01	 1.4204634e+00	 1.3203674e-01	  1.2515219e+00 	 1.7721200e-01
      79	 1.3757322e+00	 1.3118756e-01	 1.4267401e+00	 1.3127233e-01	  1.2498217e+00 	 1.9520974e-01


.. parsed-literal::

      80	 1.3841192e+00	 1.3022307e-01	 1.4354131e+00	 1.3047318e-01	  1.2431672e+00 	 2.0470047e-01


.. parsed-literal::

      81	 1.3881397e+00	 1.2987957e-01	 1.4395130e+00	 1.2994856e-01	  1.2413219e+00 	 3.2398176e-01


.. parsed-literal::

      82	 1.3922497e+00	 1.2964433e-01	 1.4436471e+00	 1.2985428e-01	  1.2387602e+00 	 2.1253872e-01
      83	 1.3978062e+00	 1.2927822e-01	 1.4493874e+00	 1.2946347e-01	  1.2347359e+00 	 1.9889975e-01


.. parsed-literal::

      84	 1.4032945e+00	 1.2900066e-01	 1.4550567e+00	 1.2896618e-01	  1.2300129e+00 	 2.0957160e-01


.. parsed-literal::

      85	 1.4088982e+00	 1.2859805e-01	 1.4607808e+00	 1.2840464e-01	  1.2371357e+00 	 2.0900059e-01
      86	 1.4130178e+00	 1.2837738e-01	 1.4649403e+00	 1.2783716e-01	  1.2480148e+00 	 1.6499639e-01


.. parsed-literal::

      87	 1.4168204e+00	 1.2799684e-01	 1.4687595e+00	 1.2730045e-01	  1.2551690e+00 	 2.1171379e-01
      88	 1.4206979e+00	 1.2777890e-01	 1.4726673e+00	 1.2695923e-01	[ 1.2595364e+00]	 1.9431901e-01


.. parsed-literal::

      89	 1.4246506e+00	 1.2765781e-01	 1.4765676e+00	 1.2664700e-01	[ 1.2640227e+00]	 2.0054245e-01


.. parsed-literal::

      90	 1.4286667e+00	 1.2728548e-01	 1.4806747e+00	 1.2613277e-01	  1.2553475e+00 	 2.1275401e-01
      91	 1.4336140e+00	 1.2719179e-01	 1.4855285e+00	 1.2570040e-01	  1.2604219e+00 	 1.8635273e-01


.. parsed-literal::

      92	 1.4365018e+00	 1.2714255e-01	 1.4884003e+00	 1.2547831e-01	  1.2618178e+00 	 2.1149373e-01
      93	 1.4400981e+00	 1.2703802e-01	 1.4920717e+00	 1.2527934e-01	  1.2553969e+00 	 1.8354607e-01


.. parsed-literal::

      94	 1.4414888e+00	 1.2687886e-01	 1.4936516e+00	 1.2528554e-01	  1.2480112e+00 	 2.0484114e-01


.. parsed-literal::

      95	 1.4458964e+00	 1.2678701e-01	 1.4979665e+00	 1.2522216e-01	  1.2466112e+00 	 2.0994616e-01


.. parsed-literal::

      96	 1.4479984e+00	 1.2663376e-01	 1.5000824e+00	 1.2522214e-01	  1.2432887e+00 	 2.1237564e-01
      97	 1.4504466e+00	 1.2642068e-01	 1.5025948e+00	 1.2520977e-01	  1.2379030e+00 	 1.6723943e-01


.. parsed-literal::

      98	 1.4545933e+00	 1.2607756e-01	 1.5068671e+00	 1.2499056e-01	  1.2265591e+00 	 1.8274641e-01
      99	 1.4559398e+00	 1.2586250e-01	 1.5086056e+00	 1.2494319e-01	  1.2043557e+00 	 1.7285848e-01


.. parsed-literal::

     100	 1.4625369e+00	 1.2566007e-01	 1.5150452e+00	 1.2458568e-01	  1.2018060e+00 	 2.0521474e-01


.. parsed-literal::

     101	 1.4644809e+00	 1.2561266e-01	 1.5168846e+00	 1.2443960e-01	  1.2063649e+00 	 2.1555519e-01
     102	 1.4680655e+00	 1.2565977e-01	 1.5205052e+00	 1.2437608e-01	  1.2031193e+00 	 1.7561960e-01


.. parsed-literal::

     103	 1.4710579e+00	 1.2570141e-01	 1.5236005e+00	 1.2391117e-01	  1.1957994e+00 	 1.7433023e-01
     104	 1.4736827e+00	 1.2563056e-01	 1.5261949e+00	 1.2391045e-01	  1.1962360e+00 	 1.9084334e-01


.. parsed-literal::

     105	 1.4770176e+00	 1.2559666e-01	 1.5295369e+00	 1.2382845e-01	  1.1992827e+00 	 2.1727967e-01


.. parsed-literal::

     106	 1.4803109e+00	 1.2539847e-01	 1.5328741e+00	 1.2359147e-01	  1.2006981e+00 	 2.0530605e-01


.. parsed-literal::

     107	 1.4843853e+00	 1.2545369e-01	 1.5370228e+00	 1.2338827e-01	  1.2099478e+00 	 2.1346426e-01


.. parsed-literal::

     108	 1.4872453e+00	 1.2533429e-01	 1.5398910e+00	 1.2320981e-01	  1.2105518e+00 	 2.1332312e-01


.. parsed-literal::

     109	 1.4898873e+00	 1.2512187e-01	 1.5425193e+00	 1.2304185e-01	  1.2142031e+00 	 2.0908356e-01
     110	 1.4931238e+00	 1.2507208e-01	 1.5457728e+00	 1.2283772e-01	  1.2135734e+00 	 1.9772100e-01


.. parsed-literal::

     111	 1.4959961e+00	 1.2505133e-01	 1.5487049e+00	 1.2286400e-01	  1.2077887e+00 	 1.8237877e-01


.. parsed-literal::

     112	 1.4985680e+00	 1.2491573e-01	 1.5512401e+00	 1.2283293e-01	  1.2051306e+00 	 2.1594429e-01


.. parsed-literal::

     113	 1.5009134e+00	 1.2493636e-01	 1.5535855e+00	 1.2283481e-01	  1.1999959e+00 	 2.0690727e-01


.. parsed-literal::

     114	 1.5038149e+00	 1.2499424e-01	 1.5565480e+00	 1.2316613e-01	  1.1955419e+00 	 2.0656252e-01


.. parsed-literal::

     115	 1.5064185e+00	 1.2474905e-01	 1.5592248e+00	 1.2292667e-01	  1.1856628e+00 	 2.1575809e-01


.. parsed-literal::

     116	 1.5083190e+00	 1.2462311e-01	 1.5610966e+00	 1.2280602e-01	  1.1873930e+00 	 2.0831943e-01


.. parsed-literal::

     117	 1.5105641e+00	 1.2435715e-01	 1.5633412e+00	 1.2261930e-01	  1.1859268e+00 	 2.0964718e-01


.. parsed-literal::

     118	 1.5126509e+00	 1.2414098e-01	 1.5654416e+00	 1.2236266e-01	  1.1789402e+00 	 2.0225215e-01


.. parsed-literal::

     119	 1.5148217e+00	 1.2397691e-01	 1.5676459e+00	 1.2226856e-01	  1.1701758e+00 	 2.0947862e-01


.. parsed-literal::

     120	 1.5168108e+00	 1.2390278e-01	 1.5697103e+00	 1.2225275e-01	  1.1541854e+00 	 2.0664954e-01


.. parsed-literal::

     121	 1.5186295e+00	 1.2394215e-01	 1.5716172e+00	 1.2235505e-01	  1.1384519e+00 	 2.0423412e-01
     122	 1.5207768e+00	 1.2398946e-01	 1.5738529e+00	 1.2244262e-01	  1.1187874e+00 	 2.0012879e-01


.. parsed-literal::

     123	 1.5229207e+00	 1.2402198e-01	 1.5760528e+00	 1.2247016e-01	  1.1056364e+00 	 2.0596862e-01


.. parsed-literal::

     124	 1.5249829e+00	 1.2405876e-01	 1.5781460e+00	 1.2238801e-01	  1.1001437e+00 	 2.2193837e-01
     125	 1.5270519e+00	 1.2393731e-01	 1.5801868e+00	 1.2223646e-01	  1.0941115e+00 	 1.8841720e-01


.. parsed-literal::

     126	 1.5290167e+00	 1.2372190e-01	 1.5821296e+00	 1.2208656e-01	  1.0915878e+00 	 2.0482635e-01


.. parsed-literal::

     127	 1.5310406e+00	 1.2353505e-01	 1.5841328e+00	 1.2212993e-01	  1.0873986e+00 	 2.1138573e-01


.. parsed-literal::

     128	 1.5329927e+00	 1.2335344e-01	 1.5860850e+00	 1.2213981e-01	  1.0881449e+00 	 2.0215964e-01
     129	 1.5343749e+00	 1.2334128e-01	 1.5874903e+00	 1.2229494e-01	  1.0844438e+00 	 1.8019152e-01


.. parsed-literal::

     130	 1.5367266e+00	 1.2339970e-01	 1.5899441e+00	 1.2267011e-01	  1.0836407e+00 	 2.1463823e-01
     131	 1.5374559e+00	 1.2328508e-01	 1.5908071e+00	 1.2253624e-01	  1.0849353e+00 	 1.9554090e-01


.. parsed-literal::

     132	 1.5392915e+00	 1.2330413e-01	 1.5925468e+00	 1.2253598e-01	  1.0862859e+00 	 2.0956278e-01
     133	 1.5401893e+00	 1.2325688e-01	 1.5934092e+00	 1.2243580e-01	  1.0880206e+00 	 1.9356942e-01


.. parsed-literal::

     134	 1.5414009e+00	 1.2316031e-01	 1.5946195e+00	 1.2230028e-01	  1.0860087e+00 	 1.8141150e-01


.. parsed-literal::

     135	 1.5427175e+00	 1.2303885e-01	 1.5959696e+00	 1.2218735e-01	  1.0776778e+00 	 2.1279144e-01
     136	 1.5444300e+00	 1.2300698e-01	 1.5976897e+00	 1.2221607e-01	  1.0677324e+00 	 1.9002056e-01


.. parsed-literal::

     137	 1.5457808e+00	 1.2299870e-01	 1.5990816e+00	 1.2232467e-01	  1.0537249e+00 	 1.8441701e-01


.. parsed-literal::

     138	 1.5470739e+00	 1.2300341e-01	 1.6004153e+00	 1.2246229e-01	  1.0398238e+00 	 2.1003819e-01


.. parsed-literal::

     139	 1.5480308e+00	 1.2317556e-01	 1.6014594e+00	 1.2272133e-01	  1.0055835e+00 	 2.1251607e-01


.. parsed-literal::

     140	 1.5496382e+00	 1.2308951e-01	 1.6030756e+00	 1.2278351e-01	  1.0060086e+00 	 2.0994902e-01


.. parsed-literal::

     141	 1.5503805e+00	 1.2305505e-01	 1.6038094e+00	 1.2277203e-01	  1.0085902e+00 	 2.1456742e-01
     142	 1.5517285e+00	 1.2299421e-01	 1.6051786e+00	 1.2279763e-01	  1.0029593e+00 	 1.9838357e-01


.. parsed-literal::

     143	 1.5534409e+00	 1.2285788e-01	 1.6069393e+00	 1.2288571e-01	  9.8573865e-01 	 1.9359279e-01
     144	 1.5551358e+00	 1.2261414e-01	 1.6087307e+00	 1.2290733e-01	  9.6199418e-01 	 1.7691612e-01


.. parsed-literal::

     145	 1.5563948e+00	 1.2251495e-01	 1.6099776e+00	 1.2285520e-01	  9.5663924e-01 	 2.0921516e-01


.. parsed-literal::

     146	 1.5575845e+00	 1.2235598e-01	 1.6112057e+00	 1.2285613e-01	  9.4785894e-01 	 2.1026826e-01


.. parsed-literal::

     147	 1.5585733e+00	 1.2220395e-01	 1.6122819e+00	 1.2293508e-01	  9.4100200e-01 	 2.0608473e-01


.. parsed-literal::

     148	 1.5597492e+00	 1.2212373e-01	 1.6135151e+00	 1.2307231e-01	  9.3716088e-01 	 2.0842791e-01
     149	 1.5607590e+00	 1.2209728e-01	 1.6145854e+00	 1.2321523e-01	  9.3159073e-01 	 2.0007396e-01


.. parsed-literal::

     150	 1.5617119e+00	 1.2206662e-01	 1.6155763e+00	 1.2326996e-01	  9.2504051e-01 	 2.1786261e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.1 s, total: 2min 4s
    Wall time: 31.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fc8a02db730>



This should have taken about 30 seconds on a typical desktop computer,
and you should now see a file called ``GPz_model.pkl`` in the directory.
This model file is used by the ``GPzEstimator`` stage to determine our
redshift PDFs for the test set of galaxies. Let’s set up that stage,
again defining a dictionary of variables for the config params:

.. code:: ipython3

    gpz_test_dict = dict(hdf5_groupname="photometry", model="GPz_model.pkl")
    
    gpz_run = GPzEstimator.make_stage(name="gpz_run", **gpz_test_dict)

Let’s run the stage and compute photo-z’s for our test set:

.. code:: ipython3

    %%time
    results = gpz_run.estimate(test_data)


.. parsed-literal::

    Inserting handle into data store.  model: GPz_model.pkl, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.17 s, sys: 43 ms, total: 2.21 s
    Wall time: 681 ms


This should be very fast, under a second for our 20,449 galaxies in the
test set. Now, let’s plot a scatter plot of the point estimates, as well
as a few example PDFs. We can get access to the ``qp`` ensemble that was
written via the DataStore via ``results()``

.. code:: ipython3

    ens = results()

.. code:: ipython3

    expdfids = [2, 180, 13517, 18032]
    fig, axs = plt.subplots(4, 1, figsize=(12,10))
    for i, xx in enumerate(expdfids):
        axs[i].set_xlim(0,3)
        ens[xx].plot_native(axes=axs[i])
    axs[3].set_xlabel("redshift", fontsize=15)




.. parsed-literal::

    Text(0.5, 0, 'redshift')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_16_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data.data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_19_1.png

