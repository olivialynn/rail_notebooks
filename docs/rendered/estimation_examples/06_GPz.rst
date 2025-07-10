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
       1	-3.5662683e-01	 3.2436880e-01	-3.4696626e-01	 3.0450976e-01	[-3.1165496e-01]	 4.4212914e-01


.. parsed-literal::

       2	-2.8319861e-01	 3.1296032e-01	-2.5844375e-01	 2.9455147e-01	[-2.0523368e-01]	 2.2777653e-01


.. parsed-literal::

       3	-2.3928712e-01	 2.9250271e-01	-1.9682665e-01	 2.7570067e-01	[-1.3248826e-01]	 2.7562785e-01
       4	-2.0313350e-01	 2.6805515e-01	-1.6163445e-01	 2.5572608e-01	[-9.4023652e-02]	 1.9259214e-01


.. parsed-literal::

       5	-1.1184431e-01	 2.5854075e-01	-7.6584237e-02	 2.4901541e-01	[-3.3243320e-02]	 2.1210861e-01
       6	-7.6474946e-02	 2.5316788e-01	-4.5740598e-02	 2.4535263e-01	[-1.8152418e-02]	 1.9681334e-01


.. parsed-literal::

       7	-5.8429338e-02	 2.5026416e-01	-3.3976892e-02	 2.4328535e-01	[-7.1176981e-03]	 2.0535159e-01


.. parsed-literal::

       8	-4.5934497e-02	 2.4817669e-01	-2.5357868e-02	 2.4165966e-01	[ 6.7650891e-04]	 2.1314597e-01


.. parsed-literal::

       9	-3.2311532e-02	 2.4563829e-01	-1.4646579e-02	 2.4000964e-01	[ 8.9747529e-03]	 2.0529199e-01
      10	-2.0474411e-02	 2.4337079e-01	-4.9522802e-03	 2.3916617e-01	[ 1.5035810e-02]	 1.9691730e-01


.. parsed-literal::

      11	-1.5968329e-02	 2.4272428e-01	-1.4826023e-03	 2.3935017e-01	  1.4359469e-02 	 2.0239544e-01
      12	-1.2683955e-02	 2.4198602e-01	 1.6263241e-03	 2.3874716e-01	[ 1.7258103e-02]	 1.7893791e-01


.. parsed-literal::

      13	-8.7067339e-03	 2.4115517e-01	 5.5578868e-03	 2.3792185e-01	[ 2.0569525e-02]	 1.9610381e-01
      14	-4.1510304e-03	 2.4010968e-01	 1.0832455e-02	 2.3697641e-01	[ 2.5830881e-02]	 1.9663286e-01


.. parsed-literal::

      15	 1.2352496e-01	 2.2605499e-01	 1.4742156e-01	 2.1977214e-01	[ 1.7142351e-01]	 3.1651139e-01


.. parsed-literal::

      16	 1.9483865e-01	 2.2522109e-01	 2.2267667e-01	 2.2021302e-01	[ 2.2556003e-01]	 2.9897904e-01


.. parsed-literal::

      17	 2.4697284e-01	 2.1822386e-01	 2.7565779e-01	 2.1522661e-01	[ 2.7275172e-01]	 2.0330501e-01


.. parsed-literal::

      18	 3.1411138e-01	 2.1299529e-01	 3.4488215e-01	 2.0936157e-01	[ 3.3890237e-01]	 2.0997190e-01


.. parsed-literal::

      19	 3.6662084e-01	 2.0904069e-01	 3.9946148e-01	 2.0711087e-01	[ 3.7247545e-01]	 2.0977521e-01


.. parsed-literal::

      20	 4.1336286e-01	 2.0760416e-01	 4.4761138e-01	 2.0827894e-01	[ 3.9548023e-01]	 2.1948910e-01


.. parsed-literal::

      21	 4.5266032e-01	 2.0489343e-01	 4.8776087e-01	 2.0658068e-01	[ 4.2774689e-01]	 2.1000767e-01
      22	 5.4632922e-01	 2.0220750e-01	 5.8361505e-01	 2.0286460e-01	[ 4.9868403e-01]	 1.9213009e-01


.. parsed-literal::

      23	 6.1819964e-01	 1.9683066e-01	 6.5816680e-01	 1.9884587e-01	[ 5.2554490e-01]	 1.8192887e-01


.. parsed-literal::

      24	 6.3672955e-01	 1.9436535e-01	 6.7406516e-01	 1.9700905e-01	[ 5.8208723e-01]	 2.1057749e-01
      25	 6.6947601e-01	 1.9411085e-01	 7.0846951e-01	 1.9757445e-01	[ 5.8858414e-01]	 1.8283725e-01


.. parsed-literal::

      26	 7.0498262e-01	 1.9453444e-01	 7.4427783e-01	 1.9669477e-01	[ 6.2749415e-01]	 2.1358848e-01


.. parsed-literal::

      27	 7.4689966e-01	 1.9480712e-01	 7.8590419e-01	 1.9468758e-01	[ 6.8412424e-01]	 2.0460296e-01


.. parsed-literal::

      28	 7.5654050e-01	 2.1592627e-01	 7.9486694e-01	 2.1143064e-01	[ 7.4811917e-01]	 2.0226622e-01
      29	 8.1223558e-01	 1.9859556e-01	 8.5293972e-01	 1.9717713e-01	[ 7.8607425e-01]	 2.0168185e-01


.. parsed-literal::

      30	 8.4141046e-01	 1.9467028e-01	 8.8160757e-01	 1.9500025e-01	[ 8.1063623e-01]	 1.7644978e-01


.. parsed-literal::

      31	 8.7001328e-01	 1.9247945e-01	 9.1112824e-01	 1.9153118e-01	[ 8.3467359e-01]	 2.1750903e-01


.. parsed-literal::

      32	 8.9247273e-01	 1.9275342e-01	 9.3413420e-01	 1.8974076e-01	[ 8.7241581e-01]	 2.0360541e-01
      33	 9.1927994e-01	 1.9099809e-01	 9.6198168e-01	 1.8717597e-01	[ 8.9752001e-01]	 1.7660379e-01


.. parsed-literal::

      34	 9.3551311e-01	 1.8759057e-01	 9.7790158e-01	 1.8458734e-01	[ 9.1758673e-01]	 2.1035194e-01
      35	 9.5138433e-01	 1.8503015e-01	 9.9431778e-01	 1.8145741e-01	[ 9.3474783e-01]	 1.7715693e-01


.. parsed-literal::

      36	 9.8000608e-01	 1.7881712e-01	 1.0247543e+00	 1.7468395e-01	[ 9.6133989e-01]	 1.9951701e-01
      37	 9.9218892e-01	 1.7499361e-01	 1.0378566e+00	 1.7143586e-01	[ 9.7436954e-01]	 1.9842839e-01


.. parsed-literal::

      38	 1.0116813e+00	 1.7099570e-01	 1.0570074e+00	 1.6974430e-01	[ 9.8846681e-01]	 2.0851326e-01
      39	 1.0220301e+00	 1.6917453e-01	 1.0673627e+00	 1.6901226e-01	[ 9.9750178e-01]	 1.9686055e-01


.. parsed-literal::

      40	 1.0349513e+00	 1.6736096e-01	 1.0804942e+00	 1.6794099e-01	[ 1.0096267e+00]	 2.0853615e-01


.. parsed-literal::

      41	 1.0519485e+00	 1.6522880e-01	 1.0979775e+00	 1.6487292e-01	[ 1.0262577e+00]	 2.0080519e-01
      42	 1.0591454e+00	 1.6444000e-01	 1.1058795e+00	 1.6269649e-01	[ 1.0265549e+00]	 1.9707847e-01


.. parsed-literal::

      43	 1.0709656e+00	 1.6304075e-01	 1.1174990e+00	 1.6073197e-01	[ 1.0393429e+00]	 1.7264628e-01


.. parsed-literal::

      44	 1.0794106e+00	 1.6173150e-01	 1.1260273e+00	 1.5946253e-01	[ 1.0465405e+00]	 2.0728874e-01
      45	 1.0913409e+00	 1.6066162e-01	 1.1380394e+00	 1.5683247e-01	[ 1.0574109e+00]	 1.9657373e-01


.. parsed-literal::

      46	 1.1043912e+00	 1.5781640e-01	 1.1514981e+00	 1.5346388e-01	[ 1.0746191e+00]	 2.0423913e-01
      47	 1.1173471e+00	 1.5674137e-01	 1.1644792e+00	 1.5149390e-01	[ 1.0865256e+00]	 1.9180632e-01


.. parsed-literal::

      48	 1.1263955e+00	 1.5594317e-01	 1.1737399e+00	 1.5051215e-01	[ 1.0944800e+00]	 2.0486212e-01


.. parsed-literal::

      49	 1.1406291e+00	 1.5446166e-01	 1.1883429e+00	 1.4920082e-01	[ 1.1075172e+00]	 2.0775914e-01


.. parsed-literal::

      50	 1.1538954e+00	 1.5209415e-01	 1.2018901e+00	 1.4620791e-01	[ 1.1202933e+00]	 2.0919371e-01
      51	 1.1673503e+00	 1.5063505e-01	 1.2152717e+00	 1.4491821e-01	[ 1.1369176e+00]	 1.9330978e-01


.. parsed-literal::

      52	 1.1759307e+00	 1.4948650e-01	 1.2240013e+00	 1.4344145e-01	[ 1.1472593e+00]	 1.9863153e-01


.. parsed-literal::

      53	 1.1879659e+00	 1.4766889e-01	 1.2362390e+00	 1.4166069e-01	[ 1.1584201e+00]	 2.1398950e-01
      54	 1.2002640e+00	 1.4659339e-01	 1.2492397e+00	 1.4040443e-01	[ 1.1587317e+00]	 1.7949557e-01


.. parsed-literal::

      55	 1.2134183e+00	 1.4548945e-01	 1.2622198e+00	 1.3890310e-01	[ 1.1761085e+00]	 2.0944142e-01


.. parsed-literal::

      56	 1.2244611e+00	 1.4469892e-01	 1.2735576e+00	 1.3838339e-01	[ 1.1825991e+00]	 2.0991421e-01
      57	 1.2388474e+00	 1.4423226e-01	 1.2884046e+00	 1.3753363e-01	[ 1.1915315e+00]	 1.9442081e-01


.. parsed-literal::

      58	 1.2499359e+00	 1.4180352e-01	 1.2998527e+00	 1.3718748e-01	[ 1.1982408e+00]	 1.9555426e-01


.. parsed-literal::

      59	 1.2603795e+00	 1.4119762e-01	 1.3103175e+00	 1.3625765e-01	[ 1.2058156e+00]	 2.0276356e-01


.. parsed-literal::

      60	 1.2704466e+00	 1.4004978e-01	 1.3205992e+00	 1.3533754e-01	[ 1.2201749e+00]	 2.1759009e-01
      61	 1.2801477e+00	 1.3913405e-01	 1.3307890e+00	 1.3449868e-01	[ 1.2265828e+00]	 1.6626549e-01


.. parsed-literal::

      62	 1.2914114e+00	 1.3822901e-01	 1.3420839e+00	 1.3424688e-01	[ 1.2387179e+00]	 2.1680665e-01
      63	 1.3036294e+00	 1.3730334e-01	 1.3545763e+00	 1.3376603e-01	[ 1.2506472e+00]	 1.9739246e-01


.. parsed-literal::

      64	 1.3136491e+00	 1.3673766e-01	 1.3647642e+00	 1.3313199e-01	[ 1.2566792e+00]	 2.0162010e-01


.. parsed-literal::

      65	 1.3232132e+00	 1.3598262e-01	 1.3743598e+00	 1.3227960e-01	[ 1.2688976e+00]	 2.0609665e-01
      66	 1.3317288e+00	 1.3578107e-01	 1.3829253e+00	 1.3228278e-01	[ 1.2732105e+00]	 1.8183017e-01


.. parsed-literal::

      67	 1.3416866e+00	 1.3554413e-01	 1.3931628e+00	 1.3230496e-01	  1.2719718e+00 	 2.0480371e-01


.. parsed-literal::

      68	 1.3512469e+00	 1.3539947e-01	 1.4027007e+00	 1.3276194e-01	  1.2715138e+00 	 2.0801377e-01


.. parsed-literal::

      69	 1.3596639e+00	 1.3541913e-01	 1.4109823e+00	 1.3323435e-01	[ 1.2742845e+00]	 2.1042085e-01


.. parsed-literal::

      70	 1.3681705e+00	 1.3482294e-01	 1.4198032e+00	 1.3318757e-01	[ 1.2762334e+00]	 2.1324492e-01


.. parsed-literal::

      71	 1.3754269e+00	 1.3471862e-01	 1.4272551e+00	 1.3307380e-01	[ 1.2817728e+00]	 2.0490766e-01
      72	 1.3837822e+00	 1.3420827e-01	 1.4358853e+00	 1.3263164e-01	[ 1.2898563e+00]	 1.8490672e-01


.. parsed-literal::

      73	 1.3910205e+00	 1.3414426e-01	 1.4432687e+00	 1.3229302e-01	[ 1.2951918e+00]	 1.8212080e-01


.. parsed-literal::

      74	 1.3971380e+00	 1.3402260e-01	 1.4495765e+00	 1.3174829e-01	[ 1.2959278e+00]	 2.0890307e-01
      75	 1.4031184e+00	 1.3411096e-01	 1.4556242e+00	 1.3197348e-01	[ 1.2990378e+00]	 1.8973517e-01


.. parsed-literal::

      76	 1.4092177e+00	 1.3382368e-01	 1.4616960e+00	 1.3165210e-01	[ 1.3049421e+00]	 2.0329094e-01


.. parsed-literal::

      77	 1.4138198e+00	 1.3320572e-01	 1.4663208e+00	 1.3108753e-01	[ 1.3067899e+00]	 2.1017241e-01


.. parsed-literal::

      78	 1.4190598e+00	 1.3260736e-01	 1.4717149e+00	 1.3096506e-01	[ 1.3103234e+00]	 2.0525932e-01
      79	 1.4249876e+00	 1.3184917e-01	 1.4775916e+00	 1.3065069e-01	[ 1.3124938e+00]	 1.8038797e-01


.. parsed-literal::

      80	 1.4304486e+00	 1.3122351e-01	 1.4831099e+00	 1.3019394e-01	  1.3115891e+00 	 2.0971441e-01
      81	 1.4349937e+00	 1.3090800e-01	 1.4877603e+00	 1.3012990e-01	[ 1.3154529e+00]	 1.8184805e-01


.. parsed-literal::

      82	 1.4393050e+00	 1.3071583e-01	 1.4921395e+00	 1.2993548e-01	[ 1.3203574e+00]	 2.0443130e-01


.. parsed-literal::

      83	 1.4444412e+00	 1.3052116e-01	 1.4974037e+00	 1.2953246e-01	[ 1.3260250e+00]	 2.1170139e-01
      84	 1.4482208e+00	 1.3030653e-01	 1.5012687e+00	 1.2915599e-01	[ 1.3299890e+00]	 1.9128180e-01


.. parsed-literal::

      85	 1.4513205e+00	 1.3014716e-01	 1.5042613e+00	 1.2898832e-01	[ 1.3323507e+00]	 1.8387985e-01


.. parsed-literal::

      86	 1.4559423e+00	 1.2972108e-01	 1.5088984e+00	 1.2856443e-01	[ 1.3329893e+00]	 2.1243095e-01
      87	 1.4597306e+00	 1.2929980e-01	 1.5127117e+00	 1.2811001e-01	[ 1.3354902e+00]	 1.9575000e-01


.. parsed-literal::

      88	 1.4631674e+00	 1.2853655e-01	 1.5163817e+00	 1.2730324e-01	  1.3297359e+00 	 2.0819807e-01


.. parsed-literal::

      89	 1.4679673e+00	 1.2836732e-01	 1.5210897e+00	 1.2700145e-01	[ 1.3375069e+00]	 2.0346522e-01


.. parsed-literal::

      90	 1.4702309e+00	 1.2831838e-01	 1.5233371e+00	 1.2688996e-01	[ 1.3406311e+00]	 2.0416021e-01
      91	 1.4739121e+00	 1.2809367e-01	 1.5271154e+00	 1.2657847e-01	[ 1.3409790e+00]	 1.9408441e-01


.. parsed-literal::

      92	 1.4760627e+00	 1.2794328e-01	 1.5293403e+00	 1.2642224e-01	  1.3400800e+00 	 3.0771589e-01
      93	 1.4781294e+00	 1.2785268e-01	 1.5314032e+00	 1.2634515e-01	  1.3396541e+00 	 1.9313955e-01


.. parsed-literal::

      94	 1.4808847e+00	 1.2765653e-01	 1.5341952e+00	 1.2603984e-01	  1.3378339e+00 	 2.0210409e-01


.. parsed-literal::

      95	 1.4826890e+00	 1.2760519e-01	 1.5360531e+00	 1.2600830e-01	  1.3400710e+00 	 2.1443510e-01
      96	 1.4849833e+00	 1.2752083e-01	 1.5383870e+00	 1.2583172e-01	  1.3408545e+00 	 1.7413187e-01


.. parsed-literal::

      97	 1.4874586e+00	 1.2740706e-01	 1.5409435e+00	 1.2548520e-01	[ 1.3425438e+00]	 2.0706105e-01
      98	 1.4895100e+00	 1.2731601e-01	 1.5431121e+00	 1.2528496e-01	[ 1.3425803e+00]	 1.9709778e-01


.. parsed-literal::

      99	 1.4919182e+00	 1.2717680e-01	 1.5455732e+00	 1.2496030e-01	[ 1.3431871e+00]	 2.0064664e-01
     100	 1.4957715e+00	 1.2692123e-01	 1.5495645e+00	 1.2435854e-01	  1.3419223e+00 	 1.8957186e-01


.. parsed-literal::

     101	 1.4973612e+00	 1.2678344e-01	 1.5511875e+00	 1.2411090e-01	  1.3411789e+00 	 3.0240011e-01
     102	 1.4994695e+00	 1.2668258e-01	 1.5532967e+00	 1.2395179e-01	  1.3419222e+00 	 2.0009303e-01


.. parsed-literal::

     103	 1.5020631e+00	 1.2657520e-01	 1.5559094e+00	 1.2373493e-01	[ 1.3433150e+00]	 2.0533705e-01


.. parsed-literal::

     104	 1.5037166e+00	 1.2648687e-01	 1.5576381e+00	 1.2374299e-01	[ 1.3447916e+00]	 2.0423770e-01
     105	 1.5059062e+00	 1.2649243e-01	 1.5597380e+00	 1.2376933e-01	[ 1.3483605e+00]	 1.9979835e-01


.. parsed-literal::

     106	 1.5075414e+00	 1.2647930e-01	 1.5613968e+00	 1.2369458e-01	[ 1.3499713e+00]	 1.8221259e-01


.. parsed-literal::

     107	 1.5088465e+00	 1.2645084e-01	 1.5627315e+00	 1.2367800e-01	  1.3495595e+00 	 2.0722842e-01
     108	 1.5105617e+00	 1.2644523e-01	 1.5645777e+00	 1.2380162e-01	[ 1.3535586e+00]	 1.7640758e-01


.. parsed-literal::

     109	 1.5124765e+00	 1.2639814e-01	 1.5665248e+00	 1.2367661e-01	  1.3488048e+00 	 2.0023799e-01
     110	 1.5136883e+00	 1.2636656e-01	 1.5676768e+00	 1.2371409e-01	  1.3487008e+00 	 1.7532086e-01


.. parsed-literal::

     111	 1.5157634e+00	 1.2631003e-01	 1.5697532e+00	 1.2373243e-01	  1.3499355e+00 	 1.7561007e-01


.. parsed-literal::

     112	 1.5176037e+00	 1.2630181e-01	 1.5716995e+00	 1.2387782e-01	  1.3477849e+00 	 2.0353603e-01
     113	 1.5197468e+00	 1.2624556e-01	 1.5738427e+00	 1.2368218e-01	  1.3484414e+00 	 1.9488478e-01


.. parsed-literal::

     114	 1.5215537e+00	 1.2624351e-01	 1.5757077e+00	 1.2354462e-01	  1.3474802e+00 	 1.9272542e-01


.. parsed-literal::

     115	 1.5230063e+00	 1.2626688e-01	 1.5772173e+00	 1.2348440e-01	  1.3457152e+00 	 2.1443129e-01


.. parsed-literal::

     116	 1.5253363e+00	 1.2642856e-01	 1.5797362e+00	 1.2366547e-01	  1.3349992e+00 	 2.1398830e-01


.. parsed-literal::

     117	 1.5276948e+00	 1.2646387e-01	 1.5820867e+00	 1.2354106e-01	  1.3378503e+00 	 2.0715785e-01
     118	 1.5288762e+00	 1.2648690e-01	 1.5832164e+00	 1.2351354e-01	  1.3398075e+00 	 1.7711329e-01


.. parsed-literal::

     119	 1.5304639e+00	 1.2650321e-01	 1.5848027e+00	 1.2334355e-01	  1.3408062e+00 	 1.9820452e-01


.. parsed-literal::

     120	 1.5320192e+00	 1.2660899e-01	 1.5863975e+00	 1.2314933e-01	  1.3408565e+00 	 2.0872474e-01
     121	 1.5335842e+00	 1.2664445e-01	 1.5879974e+00	 1.2286002e-01	  1.3389095e+00 	 1.9731450e-01


.. parsed-literal::

     122	 1.5351081e+00	 1.2664056e-01	 1.5896622e+00	 1.2234261e-01	  1.3378703e+00 	 1.8944788e-01
     123	 1.5367545e+00	 1.2662693e-01	 1.5913187e+00	 1.2222653e-01	  1.3313872e+00 	 1.9469261e-01


.. parsed-literal::

     124	 1.5376652e+00	 1.2658262e-01	 1.5921897e+00	 1.2229464e-01	  1.3328033e+00 	 1.7766953e-01
     125	 1.5391717e+00	 1.2653834e-01	 1.5937084e+00	 1.2226836e-01	  1.3312227e+00 	 1.9952178e-01


.. parsed-literal::

     126	 1.5409284e+00	 1.2646907e-01	 1.5954798e+00	 1.2223728e-01	  1.3287469e+00 	 1.9348645e-01


.. parsed-literal::

     127	 1.5427870e+00	 1.2643080e-01	 1.5974375e+00	 1.2218137e-01	  1.3263186e+00 	 2.0819974e-01
     128	 1.5444161e+00	 1.2639318e-01	 1.5990634e+00	 1.2232564e-01	  1.3187690e+00 	 1.9794226e-01


.. parsed-literal::

     129	 1.5453645e+00	 1.2637701e-01	 1.5999736e+00	 1.2242261e-01	  1.3208728e+00 	 1.9510698e-01
     130	 1.5469596e+00	 1.2641535e-01	 1.6015958e+00	 1.2266360e-01	  1.3208307e+00 	 1.9529557e-01


.. parsed-literal::

     131	 1.5482945e+00	 1.2636922e-01	 1.6030544e+00	 1.2323013e-01	  1.3176567e+00 	 1.9220233e-01


.. parsed-literal::

     132	 1.5499618e+00	 1.2645716e-01	 1.6047445e+00	 1.2351633e-01	  1.3161450e+00 	 2.0199704e-01


.. parsed-literal::

     133	 1.5514495e+00	 1.2649528e-01	 1.6062903e+00	 1.2378641e-01	  1.3132458e+00 	 2.1225190e-01
     134	 1.5523166e+00	 1.2649299e-01	 1.6071610e+00	 1.2388310e-01	  1.3152058e+00 	 1.8008566e-01


.. parsed-literal::

     135	 1.5537503e+00	 1.2641745e-01	 1.6085767e+00	 1.2399483e-01	  1.3155487e+00 	 2.0517468e-01
     136	 1.5547795e+00	 1.2633474e-01	 1.6095556e+00	 1.2393030e-01	  1.3241182e+00 	 1.9538307e-01


.. parsed-literal::

     137	 1.5558387e+00	 1.2628355e-01	 1.6105727e+00	 1.2379765e-01	  1.3236826e+00 	 2.0231104e-01


.. parsed-literal::

     138	 1.5567756e+00	 1.2621215e-01	 1.6114754e+00	 1.2360545e-01	  1.3236591e+00 	 2.0633578e-01


.. parsed-literal::

     139	 1.5578561e+00	 1.2610843e-01	 1.6125500e+00	 1.2341695e-01	  1.3227362e+00 	 2.0597219e-01


.. parsed-literal::

     140	 1.5591952e+00	 1.2593081e-01	 1.6139543e+00	 1.2302138e-01	  1.3240049e+00 	 2.0457435e-01
     141	 1.5607397e+00	 1.2577812e-01	 1.6155076e+00	 1.2299556e-01	  1.3199080e+00 	 1.8305445e-01


.. parsed-literal::

     142	 1.5616162e+00	 1.2572658e-01	 1.6163967e+00	 1.2306156e-01	  1.3197606e+00 	 2.0053744e-01
     143	 1.5629740e+00	 1.2562183e-01	 1.6178193e+00	 1.2315220e-01	  1.3184881e+00 	 1.7133689e-01


.. parsed-literal::

     144	 1.5644146e+00	 1.2542816e-01	 1.6193326e+00	 1.2317291e-01	  1.3162891e+00 	 1.7289615e-01


.. parsed-literal::

     145	 1.5658054e+00	 1.2529230e-01	 1.6208186e+00	 1.2332124e-01	  1.3118906e+00 	 2.0878768e-01
     146	 1.5669202e+00	 1.2517047e-01	 1.6218820e+00	 1.2334259e-01	  1.3111352e+00 	 1.7634153e-01


.. parsed-literal::

     147	 1.5675259e+00	 1.2515416e-01	 1.6224465e+00	 1.2328796e-01	  1.3120058e+00 	 2.0344400e-01
     148	 1.5686663e+00	 1.2507800e-01	 1.6235563e+00	 1.2330518e-01	  1.3103629e+00 	 1.9706535e-01


.. parsed-literal::

     149	 1.5696252e+00	 1.2496458e-01	 1.6245521e+00	 1.2342401e-01	  1.3072423e+00 	 1.7242980e-01
     150	 1.5707260e+00	 1.2493467e-01	 1.6256262e+00	 1.2348537e-01	  1.3045092e+00 	 1.8952918e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 1s, sys: 1.06 s, total: 2min 2s
    Wall time: 30.8 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fbf4ca70c40>



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
    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.96 s, sys: 32 ms, total: 2 s
    Wall time: 588 ms


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

