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

    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4792595e-01	 3.2198178e-01	-3.3824411e-01	 3.1779638e-01	[-3.2708475e-01]	 4.6467876e-01


.. parsed-literal::

       2	-2.7958737e-01	 3.1247964e-01	-2.5649152e-01	 3.0461247e-01	[-2.3214843e-01]	 2.2961903e-01


.. parsed-literal::

       3	-2.3270174e-01	 2.8955602e-01	-1.8961620e-01	 2.8861360e-01	[-1.6764868e-01]	 2.8013396e-01
       4	-1.9361436e-01	 2.6726062e-01	-1.5048683e-01	 2.6692208e-01	[-1.2907358e-01]	 1.7411947e-01


.. parsed-literal::

       5	-1.0455135e-01	 2.5817875e-01	-7.2411890e-02	 2.5884205e-01	[-6.2271536e-02]	 2.1052432e-01


.. parsed-literal::

       6	-7.2182690e-02	 2.5238157e-01	-4.3495915e-02	 2.5049824e-01	[-3.5483334e-02]	 2.0642161e-01


.. parsed-literal::

       7	-5.4180170e-02	 2.4950378e-01	-3.0734381e-02	 2.4846911e-01	[-2.5285611e-02]	 2.1164608e-01


.. parsed-literal::

       8	-4.0169828e-02	 2.4703471e-01	-2.0459260e-02	 2.4710252e-01	[-1.9033926e-02]	 2.0462394e-01


.. parsed-literal::

       9	-2.7340865e-02	 2.4462242e-01	-1.0136723e-02	 2.4537605e-01	[-1.1258889e-02]	 2.1888947e-01


.. parsed-literal::

      10	-1.5744198e-02	 2.4230734e-01	-4.2496759e-04	 2.4358333e-01	[-1.1947763e-03]	 2.0772123e-01
      11	-1.0865509e-02	 2.4176765e-01	 3.0494647e-03	 2.4119806e-01	[ 6.0687463e-03]	 1.8059111e-01


.. parsed-literal::

      12	-7.8250865e-03	 2.4114168e-01	 5.9334831e-03	 2.4071784e-01	[ 8.3040477e-03]	 2.0844340e-01


.. parsed-literal::

      13	-3.9408586e-03	 2.4034209e-01	 9.8870623e-03	 2.3991201e-01	[ 1.2329296e-02]	 2.1289468e-01


.. parsed-literal::

      14	 1.9205766e-01	 2.2629195e-01	 2.1916326e-01	 2.2629448e-01	[ 2.1001362e-01]	 5.5422354e-01
      15	 3.2105899e-01	 2.1841324e-01	 3.5695217e-01	 2.1680701e-01	[ 3.4653851e-01]	 1.7916155e-01


.. parsed-literal::

      16	 4.0286098e-01	 2.1141567e-01	 4.3704527e-01	 2.1122248e-01	[ 4.2576826e-01]	 2.1002984e-01


.. parsed-literal::

      17	 4.6791516e-01	 2.0496522e-01	 5.0259384e-01	 2.0731260e-01	[ 4.8405492e-01]	 2.1698713e-01
      18	 5.8459493e-01	 2.0403302e-01	 6.2220874e-01	 2.0555024e-01	[ 5.9477346e-01]	 1.9874263e-01


.. parsed-literal::

      19	 6.2557377e-01	 2.0781180e-01	 6.6374679e-01	 2.0873109e-01	[ 6.4760648e-01]	 2.0456839e-01
      20	 6.6911237e-01	 2.0045629e-01	 7.0568518e-01	 2.0406583e-01	[ 6.7971153e-01]	 1.9967604e-01


.. parsed-literal::

      21	 6.9235484e-01	 1.9757421e-01	 7.3060747e-01	 2.0155315e-01	[ 7.0370104e-01]	 2.0137501e-01


.. parsed-literal::

      22	 7.0973530e-01	 2.0321023e-01	 7.4881529e-01	 2.0830270e-01	[ 7.1860788e-01]	 2.1605015e-01


.. parsed-literal::

      23	 7.6281016e-01	 1.9520744e-01	 8.0300316e-01	 2.0091781e-01	[ 7.7860746e-01]	 2.1135688e-01


.. parsed-literal::

      24	 8.0369862e-01	 1.9057711e-01	 8.4325501e-01	 1.9771952e-01	[ 8.2322566e-01]	 2.1695542e-01


.. parsed-literal::

      25	 8.3842832e-01	 1.8754058e-01	 8.7818296e-01	 1.9479401e-01	[ 8.6568424e-01]	 2.1343470e-01


.. parsed-literal::

      26	 8.7300294e-01	 1.8706929e-01	 9.1333135e-01	 1.8876737e-01	[ 9.0183338e-01]	 2.1146369e-01


.. parsed-literal::

      27	 8.8818405e-01	 1.8456269e-01	 9.2994736e-01	 1.8596600e-01	[ 9.2448251e-01]	 2.0895624e-01


.. parsed-literal::

      28	 9.1141414e-01	 1.8343813e-01	 9.5290147e-01	 1.8501865e-01	[ 9.3967895e-01]	 2.1437144e-01


.. parsed-literal::

      29	 9.2661558e-01	 1.8170116e-01	 9.6899276e-01	 1.8379441e-01	[ 9.5130338e-01]	 2.0208502e-01


.. parsed-literal::

      30	 9.4570621e-01	 1.7967343e-01	 9.8924875e-01	 1.8117952e-01	[ 9.6834818e-01]	 2.1748900e-01


.. parsed-literal::

      31	 9.6741274e-01	 1.7825924e-01	 1.0120686e+00	 1.7956729e-01	[ 9.8073043e-01]	 2.0927739e-01
      32	 9.9249714e-01	 1.7878946e-01	 1.0376729e+00	 1.8105157e-01	[ 9.9834043e-01]	 1.7339253e-01


.. parsed-literal::

      33	 1.0105199e+00	 1.7710679e-01	 1.0563721e+00	 1.7894129e-01	[ 1.0103321e+00]	 2.0644641e-01


.. parsed-literal::

      34	 1.0233180e+00	 1.7713910e-01	 1.0695062e+00	 1.7818325e-01	[ 1.0227918e+00]	 2.1195626e-01
      35	 1.0369990e+00	 1.7632354e-01	 1.0837166e+00	 1.7729752e-01	[ 1.0349604e+00]	 1.7554951e-01


.. parsed-literal::

      36	 1.0504074e+00	 1.7464368e-01	 1.0980107e+00	 1.7610940e-01	[ 1.0474777e+00]	 2.1297693e-01
      37	 1.0627174e+00	 1.7268486e-01	 1.1100197e+00	 1.7417455e-01	[ 1.0603676e+00]	 1.7400336e-01


.. parsed-literal::

      38	 1.0722906e+00	 1.7138764e-01	 1.1190804e+00	 1.7232245e-01	[ 1.0719298e+00]	 1.8381071e-01
      39	 1.0836003e+00	 1.6946707e-01	 1.1304162e+00	 1.7023009e-01	[ 1.0843065e+00]	 1.9452047e-01


.. parsed-literal::

      40	 1.1030749e+00	 1.6644991e-01	 1.1500552e+00	 1.6763302e-01	[ 1.1008793e+00]	 2.2006989e-01


.. parsed-literal::

      41	 1.1115091e+00	 1.6524272e-01	 1.1594248e+00	 1.6587141e-01	[ 1.1075993e+00]	 2.2761989e-01
      42	 1.1274784e+00	 1.6466950e-01	 1.1753116e+00	 1.6554309e-01	[ 1.1202950e+00]	 1.8125129e-01


.. parsed-literal::

      43	 1.1374991e+00	 1.6402354e-01	 1.1856471e+00	 1.6543293e-01	[ 1.1247617e+00]	 2.0834732e-01


.. parsed-literal::

      44	 1.1493778e+00	 1.6333154e-01	 1.1979742e+00	 1.6528645e-01	[ 1.1277233e+00]	 2.1978569e-01
      45	 1.1616159e+00	 1.6183838e-01	 1.2109801e+00	 1.6413084e-01	[ 1.1332840e+00]	 1.8234563e-01


.. parsed-literal::

      46	 1.1741987e+00	 1.6144251e-01	 1.2235332e+00	 1.6371750e-01	[ 1.1438933e+00]	 1.7717004e-01


.. parsed-literal::

      47	 1.1841620e+00	 1.6088348e-01	 1.2334043e+00	 1.6276204e-01	[ 1.1532375e+00]	 2.1414590e-01


.. parsed-literal::

      48	 1.1990863e+00	 1.6032060e-01	 1.2487349e+00	 1.6133197e-01	[ 1.1658290e+00]	 2.1028614e-01


.. parsed-literal::

      49	 1.2123968e+00	 1.5808022e-01	 1.2615729e+00	 1.5950294e-01	[ 1.1764339e+00]	 2.1698046e-01


.. parsed-literal::

      50	 1.2255383e+00	 1.5703926e-01	 1.2751531e+00	 1.5855507e-01	[ 1.1857807e+00]	 2.1237636e-01


.. parsed-literal::

      51	 1.2368580e+00	 1.5643193e-01	 1.2870875e+00	 1.5839446e-01	[ 1.1880465e+00]	 2.1179080e-01


.. parsed-literal::

      52	 1.2457020e+00	 1.5548368e-01	 1.2961180e+00	 1.5815888e-01	[ 1.1920996e+00]	 2.0078206e-01
      53	 1.2557308e+00	 1.5539182e-01	 1.3063289e+00	 1.5813611e-01	[ 1.1979816e+00]	 1.6867781e-01


.. parsed-literal::

      54	 1.2642495e+00	 1.5474652e-01	 1.3149035e+00	 1.5739285e-01	[ 1.2048175e+00]	 2.1276045e-01
      55	 1.2764669e+00	 1.5397250e-01	 1.3276265e+00	 1.5657369e-01	[ 1.2059157e+00]	 1.8315840e-01


.. parsed-literal::

      56	 1.2890358e+00	 1.5259977e-01	 1.3402531e+00	 1.5554478e-01	[ 1.2182857e+00]	 2.1840620e-01


.. parsed-literal::

      57	 1.2996926e+00	 1.5130792e-01	 1.3515914e+00	 1.5488824e-01	  1.2171633e+00 	 2.0866609e-01


.. parsed-literal::

      58	 1.3088774e+00	 1.5051237e-01	 1.3606292e+00	 1.5492690e-01	[ 1.2246382e+00]	 2.1048403e-01


.. parsed-literal::

      59	 1.3170671e+00	 1.4956729e-01	 1.3686665e+00	 1.5482918e-01	[ 1.2283968e+00]	 2.1482491e-01


.. parsed-literal::

      60	 1.3251464e+00	 1.4892432e-01	 1.3770205e+00	 1.5525144e-01	  1.2280657e+00 	 2.0392704e-01


.. parsed-literal::

      61	 1.3339989e+00	 1.4799611e-01	 1.3858653e+00	 1.5444904e-01	[ 1.2342168e+00]	 2.1259570e-01


.. parsed-literal::

      62	 1.3435264e+00	 1.4702586e-01	 1.3956484e+00	 1.5383323e-01	  1.2336314e+00 	 2.1417403e-01


.. parsed-literal::

      63	 1.3501483e+00	 1.4594746e-01	 1.4027264e+00	 1.5208696e-01	  1.2327357e+00 	 2.0991373e-01


.. parsed-literal::

      64	 1.3571099e+00	 1.4532985e-01	 1.4096203e+00	 1.5126121e-01	[ 1.2373522e+00]	 2.1882176e-01


.. parsed-literal::

      65	 1.3634477e+00	 1.4452295e-01	 1.4159881e+00	 1.5027390e-01	[ 1.2425955e+00]	 2.1319699e-01


.. parsed-literal::

      66	 1.3687613e+00	 1.4363066e-01	 1.4211429e+00	 1.4959718e-01	[ 1.2428134e+00]	 2.1540546e-01


.. parsed-literal::

      67	 1.3739903e+00	 1.4307515e-01	 1.4264235e+00	 1.4907432e-01	[ 1.2473566e+00]	 2.0552135e-01


.. parsed-literal::

      68	 1.3775876e+00	 1.4273107e-01	 1.4300731e+00	 1.4910525e-01	[ 1.2485733e+00]	 2.1297836e-01


.. parsed-literal::

      69	 1.3843318e+00	 1.4200487e-01	 1.4370987e+00	 1.4934275e-01	[ 1.2491487e+00]	 2.1402383e-01


.. parsed-literal::

      70	 1.3876888e+00	 1.4151901e-01	 1.4406821e+00	 1.4932633e-01	  1.2372897e+00 	 2.1048546e-01


.. parsed-literal::

      71	 1.3927303e+00	 1.4137008e-01	 1.4456229e+00	 1.4925429e-01	[ 1.2506090e+00]	 2.0288968e-01


.. parsed-literal::

      72	 1.3963804e+00	 1.4115494e-01	 1.4493011e+00	 1.4887318e-01	[ 1.2578512e+00]	 2.1157575e-01


.. parsed-literal::

      73	 1.3993198e+00	 1.4089268e-01	 1.4523143e+00	 1.4838911e-01	[ 1.2601954e+00]	 2.1416473e-01
      74	 1.4054885e+00	 1.4024802e-01	 1.4587974e+00	 1.4717822e-01	[ 1.2648136e+00]	 1.8946218e-01


.. parsed-literal::

      75	 1.4085322e+00	 1.3976636e-01	 1.4618602e+00	 1.4642502e-01	  1.2640294e+00 	 3.2637811e-01


.. parsed-literal::

      76	 1.4118199e+00	 1.3943047e-01	 1.4651159e+00	 1.4594830e-01	[ 1.2666960e+00]	 2.1084547e-01
      77	 1.4167989e+00	 1.3882459e-01	 1.4701424e+00	 1.4534406e-01	[ 1.2724155e+00]	 1.8881750e-01


.. parsed-literal::

      78	 1.4213115e+00	 1.3832093e-01	 1.4746083e+00	 1.4500854e-01	[ 1.2793237e+00]	 2.1663213e-01


.. parsed-literal::

      79	 1.4257943e+00	 1.3779498e-01	 1.4790220e+00	 1.4442492e-01	[ 1.2868739e+00]	 2.1664572e-01


.. parsed-literal::

      80	 1.4290123e+00	 1.3752677e-01	 1.4822580e+00	 1.4443422e-01	[ 1.2893969e+00]	 2.0478940e-01


.. parsed-literal::

      81	 1.4322516e+00	 1.3723700e-01	 1.4855785e+00	 1.4438275e-01	  1.2874526e+00 	 2.1637321e-01


.. parsed-literal::

      82	 1.4355252e+00	 1.3695080e-01	 1.4889458e+00	 1.4436488e-01	  1.2822339e+00 	 2.1770930e-01
      83	 1.4398344e+00	 1.3654438e-01	 1.4934312e+00	 1.4400869e-01	  1.2722719e+00 	 1.7988753e-01


.. parsed-literal::

      84	 1.4426611e+00	 1.3623180e-01	 1.4962634e+00	 1.4386349e-01	  1.2663932e+00 	 2.7801609e-01


.. parsed-literal::

      85	 1.4452101e+00	 1.3616228e-01	 1.4987709e+00	 1.4359806e-01	  1.2697547e+00 	 2.1578097e-01


.. parsed-literal::

      86	 1.4482412e+00	 1.3601887e-01	 1.5018131e+00	 1.4326140e-01	  1.2741236e+00 	 2.1073699e-01


.. parsed-literal::

      87	 1.4522049e+00	 1.3591545e-01	 1.5058772e+00	 1.4279210e-01	  1.2680308e+00 	 2.1184301e-01


.. parsed-literal::

      88	 1.4557486e+00	 1.3551348e-01	 1.5094452e+00	 1.4246124e-01	  1.2734284e+00 	 2.0882535e-01


.. parsed-literal::

      89	 1.4583093e+00	 1.3531910e-01	 1.5120072e+00	 1.4237885e-01	  1.2658080e+00 	 2.2136164e-01


.. parsed-literal::

      90	 1.4614428e+00	 1.3516768e-01	 1.5152308e+00	 1.4249100e-01	  1.2560307e+00 	 2.0365930e-01


.. parsed-literal::

      91	 1.4637965e+00	 1.3480169e-01	 1.5176678e+00	 1.4243565e-01	  1.2290349e+00 	 2.1422100e-01
      92	 1.4662249e+00	 1.3467897e-01	 1.5200313e+00	 1.4248359e-01	  1.2331732e+00 	 1.7713571e-01


.. parsed-literal::

      93	 1.4682334e+00	 1.3453026e-01	 1.5220260e+00	 1.4239192e-01	  1.2386343e+00 	 1.8488550e-01
      94	 1.4698121e+00	 1.3437214e-01	 1.5236196e+00	 1.4224895e-01	  1.2341562e+00 	 1.7261028e-01


.. parsed-literal::

      95	 1.4730507e+00	 1.3410064e-01	 1.5268794e+00	 1.4180126e-01	  1.2246849e+00 	 2.1625924e-01
      96	 1.4759395e+00	 1.3377796e-01	 1.5298933e+00	 1.4125342e-01	  1.2146650e+00 	 2.0758247e-01


.. parsed-literal::

      97	 1.4786258e+00	 1.3357496e-01	 1.5325977e+00	 1.4082930e-01	  1.2013967e+00 	 2.1302772e-01
      98	 1.4805861e+00	 1.3345254e-01	 1.5345734e+00	 1.4052603e-01	  1.1998519e+00 	 1.9823265e-01


.. parsed-literal::

      99	 1.4823048e+00	 1.3332664e-01	 1.5363401e+00	 1.4027308e-01	  1.1921715e+00 	 2.2136116e-01


.. parsed-literal::

     100	 1.4841740e+00	 1.3308599e-01	 1.5382287e+00	 1.3992694e-01	  1.1919547e+00 	 2.0727277e-01


.. parsed-literal::

     101	 1.4860590e+00	 1.3285045e-01	 1.5401318e+00	 1.3963365e-01	  1.1889587e+00 	 2.1017599e-01


.. parsed-literal::

     102	 1.4883010e+00	 1.3242011e-01	 1.5424190e+00	 1.3917856e-01	  1.1840030e+00 	 2.0865250e-01


.. parsed-literal::

     103	 1.4901574e+00	 1.3228029e-01	 1.5442967e+00	 1.3885602e-01	  1.1842563e+00 	 2.0762491e-01


.. parsed-literal::

     104	 1.4917741e+00	 1.3224895e-01	 1.5458826e+00	 1.3881521e-01	  1.1876635e+00 	 2.2316432e-01
     105	 1.4936758e+00	 1.3223323e-01	 1.5478016e+00	 1.3878477e-01	  1.1875742e+00 	 2.0419645e-01


.. parsed-literal::

     106	 1.4956552e+00	 1.3218134e-01	 1.5498608e+00	 1.3867147e-01	  1.1808311e+00 	 2.0866609e-01
     107	 1.4977260e+00	 1.3222872e-01	 1.5520529e+00	 1.3873931e-01	  1.1724797e+00 	 1.8157244e-01


.. parsed-literal::

     108	 1.4994093e+00	 1.3225035e-01	 1.5537155e+00	 1.3868220e-01	  1.1633515e+00 	 1.8163848e-01
     109	 1.5012427e+00	 1.3228835e-01	 1.5555135e+00	 1.3868113e-01	  1.1601343e+00 	 1.8410683e-01


.. parsed-literal::

     110	 1.5040157e+00	 1.3226980e-01	 1.5583016e+00	 1.3855676e-01	  1.1574430e+00 	 2.1063042e-01


.. parsed-literal::

     111	 1.5058988e+00	 1.3242747e-01	 1.5603815e+00	 1.3821304e-01	  1.1610542e+00 	 2.0783758e-01


.. parsed-literal::

     112	 1.5090280e+00	 1.3221319e-01	 1.5634472e+00	 1.3809977e-01	  1.1637560e+00 	 2.0088220e-01
     113	 1.5104557e+00	 1.3208072e-01	 1.5648995e+00	 1.3790147e-01	  1.1698350e+00 	 1.8770576e-01


.. parsed-literal::

     114	 1.5122219e+00	 1.3181937e-01	 1.5667864e+00	 1.3759545e-01	  1.1695312e+00 	 2.1113133e-01


.. parsed-literal::

     115	 1.5140977e+00	 1.3163686e-01	 1.5687641e+00	 1.3717193e-01	  1.1763131e+00 	 2.1376133e-01


.. parsed-literal::

     116	 1.5159415e+00	 1.3158120e-01	 1.5706445e+00	 1.3700152e-01	  1.1746405e+00 	 2.1672392e-01
     117	 1.5174130e+00	 1.3137136e-01	 1.5721336e+00	 1.3685983e-01	  1.1668727e+00 	 1.7997479e-01


.. parsed-literal::

     118	 1.5186614e+00	 1.3145648e-01	 1.5733602e+00	 1.3686261e-01	  1.1707139e+00 	 2.1998715e-01


.. parsed-literal::

     119	 1.5196430e+00	 1.3145435e-01	 1.5743079e+00	 1.3682236e-01	  1.1737580e+00 	 2.2415257e-01


.. parsed-literal::

     120	 1.5222718e+00	 1.3140564e-01	 1.5769165e+00	 1.3653948e-01	  1.1773608e+00 	 2.1103525e-01


.. parsed-literal::

     121	 1.5233554e+00	 1.3132841e-01	 1.5781989e+00	 1.3618677e-01	  1.1809589e+00 	 2.1041727e-01


.. parsed-literal::

     122	 1.5252784e+00	 1.3125527e-01	 1.5800635e+00	 1.3601210e-01	  1.1780809e+00 	 2.0734286e-01
     123	 1.5268680e+00	 1.3108573e-01	 1.5817345e+00	 1.3566297e-01	  1.1750623e+00 	 1.9589186e-01


.. parsed-literal::

     124	 1.5284102e+00	 1.3090385e-01	 1.5833976e+00	 1.3525326e-01	  1.1750289e+00 	 2.0664048e-01


.. parsed-literal::

     125	 1.5312786e+00	 1.3052256e-01	 1.5864059e+00	 1.3452778e-01	  1.1794181e+00 	 2.1522093e-01


.. parsed-literal::

     126	 1.5324288e+00	 1.3046423e-01	 1.5876012e+00	 1.3408662e-01	  1.1836228e+00 	 3.3077884e-01


.. parsed-literal::

     127	 1.5343379e+00	 1.3024507e-01	 1.5895066e+00	 1.3372557e-01	  1.1888592e+00 	 2.2024989e-01


.. parsed-literal::

     128	 1.5356887e+00	 1.3011819e-01	 1.5908015e+00	 1.3352288e-01	  1.1932710e+00 	 2.1188354e-01


.. parsed-literal::

     129	 1.5374218e+00	 1.2996802e-01	 1.5924703e+00	 1.3325672e-01	  1.1921270e+00 	 2.0889688e-01
     130	 1.5388429e+00	 1.2979821e-01	 1.5938978e+00	 1.3285902e-01	  1.1949028e+00 	 1.8021107e-01


.. parsed-literal::

     131	 1.5398468e+00	 1.2977200e-01	 1.5949354e+00	 1.3276026e-01	  1.1940895e+00 	 2.0588279e-01
     132	 1.5412523e+00	 1.2963689e-01	 1.5964853e+00	 1.3249156e-01	  1.1855233e+00 	 2.0510721e-01


.. parsed-literal::

     133	 1.5423956e+00	 1.2958517e-01	 1.5976875e+00	 1.3240333e-01	  1.1872043e+00 	 2.0699739e-01


.. parsed-literal::

     134	 1.5435960e+00	 1.2950140e-01	 1.5989285e+00	 1.3232927e-01	  1.1875402e+00 	 2.0494509e-01
     135	 1.5450425e+00	 1.2940285e-01	 1.6003779e+00	 1.3222847e-01	  1.1870296e+00 	 1.8667459e-01


.. parsed-literal::

     136	 1.5456485e+00	 1.2930538e-01	 1.6010301e+00	 1.3210601e-01	  1.1851124e+00 	 1.9176507e-01


.. parsed-literal::

     137	 1.5473284e+00	 1.2926730e-01	 1.6025854e+00	 1.3205384e-01	  1.1862852e+00 	 2.0906997e-01


.. parsed-literal::

     138	 1.5481388e+00	 1.2926108e-01	 1.6033315e+00	 1.3199839e-01	  1.1865850e+00 	 2.1535802e-01


.. parsed-literal::

     139	 1.5490602e+00	 1.2923325e-01	 1.6042104e+00	 1.3191820e-01	  1.1865126e+00 	 2.1070743e-01
     140	 1.5503035e+00	 1.2927783e-01	 1.6054716e+00	 1.3179403e-01	  1.1851141e+00 	 2.0112133e-01


.. parsed-literal::

     141	 1.5516362e+00	 1.2912404e-01	 1.6068616e+00	 1.3167788e-01	  1.1803619e+00 	 1.8608546e-01
     142	 1.5527134e+00	 1.2901582e-01	 1.6080219e+00	 1.3148854e-01	  1.1797951e+00 	 2.0250273e-01


.. parsed-literal::

     143	 1.5541158e+00	 1.2886531e-01	 1.6095634e+00	 1.3127725e-01	  1.1734854e+00 	 1.8893600e-01


.. parsed-literal::

     144	 1.5552816e+00	 1.2867386e-01	 1.6107770e+00	 1.3109060e-01	  1.1693386e+00 	 2.1009922e-01


.. parsed-literal::

     145	 1.5566822e+00	 1.2850132e-01	 1.6122084e+00	 1.3087491e-01	  1.1673049e+00 	 2.0993114e-01
     146	 1.5576371e+00	 1.2841047e-01	 1.6131584e+00	 1.3076297e-01	  1.1567711e+00 	 2.1072865e-01


.. parsed-literal::

     147	 1.5584860e+00	 1.2841376e-01	 1.6139492e+00	 1.3075721e-01	  1.1609924e+00 	 2.0191240e-01


.. parsed-literal::

     148	 1.5600458e+00	 1.2836781e-01	 1.6154766e+00	 1.3068723e-01	  1.1635360e+00 	 2.0128870e-01


.. parsed-literal::

     149	 1.5613230e+00	 1.2827428e-01	 1.6167818e+00	 1.3060442e-01	  1.1613105e+00 	 2.1626592e-01


.. parsed-literal::

     150	 1.5623644e+00	 1.2811167e-01	 1.6178709e+00	 1.3048529e-01	  1.1541363e+00 	 3.2230854e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.17 s, total: 2min 7s
    Wall time: 32.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f36d430bf10>



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
    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.77 s, sys: 62.9 ms, total: 1.84 s
    Wall time: 584 ms


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

