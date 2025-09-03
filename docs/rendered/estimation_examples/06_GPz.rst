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
       1	-3.5092792e-01	 3.2307645e-01	-3.4120647e-01	 3.1060545e-01	[-3.1839489e-01]	 4.7085857e-01


.. parsed-literal::

       2	-2.7988406e-01	 3.1210329e-01	-2.5557991e-01	 3.0069246e-01	[-2.2204949e-01]	 2.3170233e-01


.. parsed-literal::

       3	-2.3541233e-01	 2.9119707e-01	-1.9285590e-01	 2.8179099e-01	[-1.5542484e-01]	 2.9440260e-01
       4	-2.0079450e-01	 2.6720058e-01	-1.6006682e-01	 2.5866428e-01	[-1.1299096e-01]	 1.8181181e-01


.. parsed-literal::

       5	-1.0569052e-01	 2.5726452e-01	-7.0560358e-02	 2.5105162e-01	[-4.2170605e-02]	 2.1106243e-01
       6	-7.3827120e-02	 2.5277000e-01	-4.3549663e-02	 2.4752458e-01	[-2.4889577e-02]	 1.9914007e-01


.. parsed-literal::

       7	-5.3924379e-02	 2.4940444e-01	-3.0445662e-02	 2.4476344e-01	[-1.2550533e-02]	 1.9974637e-01


.. parsed-literal::

       8	-4.3216903e-02	 2.4763057e-01	-2.3111979e-02	 2.4317796e-01	[-5.6508785e-03]	 2.1357131e-01
       9	-3.0887444e-02	 2.4537290e-01	-1.3472793e-02	 2.4117584e-01	[ 3.2975672e-03]	 2.0075941e-01


.. parsed-literal::

      10	-1.7830902e-02	 2.4265606e-01	-2.0020914e-03	 2.3932084e-01	[ 1.0789461e-02]	 1.7632031e-01


.. parsed-literal::

      11	-1.4399572e-02	 2.4199507e-01	 7.8380007e-04	 2.3976025e-01	  1.0119367e-02 	 3.3176589e-01


.. parsed-literal::

      12	-1.0892414e-02	 2.4147234e-01	 3.7791375e-03	 2.4009273e-01	  9.2022171e-03 	 2.1240854e-01


.. parsed-literal::

      13	-6.7628274e-03	 2.4073388e-01	 7.5631489e-03	 2.3966981e-01	[ 1.1349876e-02]	 2.1132612e-01


.. parsed-literal::

      14	-7.8174687e-04	 2.3959603e-01	 1.3863787e-02	 2.3921750e-01	[ 1.4859983e-02]	 2.0755076e-01


.. parsed-literal::

      15	 7.6759365e-02	 2.2666045e-01	 9.6854408e-02	 2.2629649e-01	[ 9.7126062e-02]	 3.2887959e-01


.. parsed-literal::

      16	 9.6452641e-02	 2.2229783e-01	 1.1659765e-01	 2.2348517e-01	[ 1.1173702e-01]	 3.1932044e-01


.. parsed-literal::

      17	 1.5698090e-01	 2.1763931e-01	 1.7964774e-01	 2.1906909e-01	[ 1.7237067e-01]	 2.1038437e-01
      18	 2.5633656e-01	 2.1717684e-01	 2.8965533e-01	 2.1866154e-01	[ 2.6422922e-01]	 1.8300962e-01


.. parsed-literal::

      19	 3.0754022e-01	 2.1361766e-01	 3.4015175e-01	 2.1706157e-01	[ 3.0917568e-01]	 2.0666313e-01
      20	 3.4993365e-01	 2.0871206e-01	 3.8280328e-01	 2.1368196e-01	[ 3.4939627e-01]	 2.0102048e-01


.. parsed-literal::

      21	 3.9706611e-01	 2.0568427e-01	 4.2809886e-01	 2.0958927e-01	[ 3.9489867e-01]	 1.9762969e-01
      22	 4.7231920e-01	 2.0347651e-01	 5.0451272e-01	 2.0822168e-01	[ 4.6473229e-01]	 1.9179058e-01


.. parsed-literal::

      23	 5.5406508e-01	 2.0202417e-01	 5.8931920e-01	 2.0716242e-01	[ 5.4373729e-01]	 2.1174788e-01


.. parsed-literal::

      24	 6.0438243e-01	 2.0375513e-01	 6.4246233e-01	 2.0697045e-01	[ 6.0065439e-01]	 2.1312451e-01
      25	 6.2892068e-01	 2.0208292e-01	 6.6646430e-01	 2.0460384e-01	[ 6.3317529e-01]	 1.9879961e-01


.. parsed-literal::

      26	 6.4978356e-01	 1.9939466e-01	 6.8695221e-01	 2.0290432e-01	[ 6.4665752e-01]	 2.1355462e-01


.. parsed-literal::

      27	 6.7855977e-01	 1.9798680e-01	 7.1441523e-01	 2.0287183e-01	[ 6.6871555e-01]	 2.1482801e-01
      28	 7.0862655e-01	 1.9713969e-01	 7.4359741e-01	 2.0033268e-01	[ 7.0730627e-01]	 1.9148946e-01


.. parsed-literal::

      29	 7.3889322e-01	 1.9755944e-01	 7.7510294e-01	 1.9922037e-01	[ 7.3847571e-01]	 2.1709776e-01


.. parsed-literal::

      30	 7.5774028e-01	 1.9858283e-01	 7.9431747e-01	 1.9954042e-01	[ 7.6126033e-01]	 3.3007169e-01
      31	 7.7891345e-01	 2.0395387e-01	 8.1854035e-01	 2.0288212e-01	[ 7.8068530e-01]	 1.9999528e-01


.. parsed-literal::

      32	 8.0371904e-01	 2.0260161e-01	 8.4313380e-01	 2.0302094e-01	[ 8.1451199e-01]	 1.9639492e-01


.. parsed-literal::

      33	 8.2145294e-01	 1.9885669e-01	 8.6113648e-01	 2.0091651e-01	[ 8.2707366e-01]	 2.1255350e-01


.. parsed-literal::

      34	 8.4381348e-01	 1.9910557e-01	 8.8355481e-01	 2.0162092e-01	[ 8.5520838e-01]	 2.1639633e-01


.. parsed-literal::

      35	 8.8551775e-01	 1.9487930e-01	 9.2774050e-01	 1.9697558e-01	[ 9.0394509e-01]	 2.0502758e-01


.. parsed-literal::

      36	 9.0708057e-01	 1.9672021e-01	 9.4973710e-01	 1.9534297e-01	[ 9.3514550e-01]	 2.0448351e-01
      37	 9.2776345e-01	 1.9292598e-01	 9.7039393e-01	 1.9246037e-01	[ 9.5682528e-01]	 1.9383621e-01


.. parsed-literal::

      38	 9.3839475e-01	 1.9060506e-01	 9.8141660e-01	 1.9113445e-01	[ 9.6459474e-01]	 2.1430516e-01


.. parsed-literal::

      39	 9.5043882e-01	 1.8920582e-01	 9.9375769e-01	 1.9051213e-01	[ 9.7482322e-01]	 2.1268177e-01
      40	 9.6274244e-01	 1.8843364e-01	 1.0064516e+00	 1.8957182e-01	[ 9.8628660e-01]	 1.8545437e-01


.. parsed-literal::

      41	 9.7326977e-01	 1.8700805e-01	 1.0175579e+00	 1.8754333e-01	[ 9.9602018e-01]	 2.1417165e-01


.. parsed-literal::

      42	 9.8303393e-01	 1.8613018e-01	 1.0272774e+00	 1.8624454e-01	[ 1.0064147e+00]	 2.1431565e-01


.. parsed-literal::

      43	 9.9229887e-01	 1.8503491e-01	 1.0367933e+00	 1.8464601e-01	[ 1.0166635e+00]	 2.1217155e-01
      44	 1.0050953e+00	 1.8364586e-01	 1.0501553e+00	 1.8265279e-01	[ 1.0314711e+00]	 1.9249153e-01


.. parsed-literal::

      45	 1.0159146e+00	 1.8281336e-01	 1.0621345e+00	 1.8164042e-01	[ 1.0413429e+00]	 1.8560553e-01
      46	 1.0292849e+00	 1.8176378e-01	 1.0754881e+00	 1.8069834e-01	[ 1.0547642e+00]	 1.8153405e-01


.. parsed-literal::

      47	 1.0375225e+00	 1.8144871e-01	 1.0834714e+00	 1.8070944e-01	[ 1.0605823e+00]	 2.1318507e-01


.. parsed-literal::

      48	 1.0494726e+00	 1.8039356e-01	 1.0955920e+00	 1.8002443e-01	[ 1.0668742e+00]	 2.0478821e-01
      49	 1.0561932e+00	 1.8028754e-01	 1.1023502e+00	 1.8016836e-01	[ 1.0716753e+00]	 1.8041945e-01


.. parsed-literal::

      50	 1.0636830e+00	 1.7906747e-01	 1.1095605e+00	 1.7899351e-01	[ 1.0775910e+00]	 1.9555330e-01
      51	 1.0711378e+00	 1.7738586e-01	 1.1170316e+00	 1.7738516e-01	[ 1.0804515e+00]	 1.7697835e-01


.. parsed-literal::

      52	 1.0790711e+00	 1.7565737e-01	 1.1251543e+00	 1.7565984e-01	[ 1.0825152e+00]	 2.1294212e-01


.. parsed-literal::

      53	 1.0893035e+00	 1.7371761e-01	 1.1359237e+00	 1.7360490e-01	[ 1.0826165e+00]	 2.1359944e-01


.. parsed-literal::

      54	 1.0977494e+00	 1.7119957e-01	 1.1446224e+00	 1.7144568e-01	[ 1.0856390e+00]	 2.0232010e-01


.. parsed-literal::

      55	 1.1041894e+00	 1.7098461e-01	 1.1511092e+00	 1.7123507e-01	[ 1.0917741e+00]	 2.1183610e-01
      56	 1.1151718e+00	 1.6923991e-01	 1.1626677e+00	 1.7020042e-01	[ 1.0929261e+00]	 1.9709277e-01


.. parsed-literal::

      57	 1.1218301e+00	 1.6882028e-01	 1.1695919e+00	 1.6972746e-01	[ 1.1024769e+00]	 1.8529153e-01
      58	 1.1292291e+00	 1.6763151e-01	 1.1769631e+00	 1.6890466e-01	[ 1.1068496e+00]	 1.9281507e-01


.. parsed-literal::

      59	 1.1419321e+00	 1.6542931e-01	 1.1900261e+00	 1.6744224e-01	[ 1.1118518e+00]	 1.9646072e-01
      60	 1.1496002e+00	 1.6463283e-01	 1.1980792e+00	 1.6675533e-01	[ 1.1182632e+00]	 1.9729567e-01


.. parsed-literal::

      61	 1.1602540e+00	 1.6437043e-01	 1.2091273e+00	 1.6602171e-01	[ 1.1294157e+00]	 2.0030999e-01
      62	 1.1695399e+00	 1.6276206e-01	 1.2185895e+00	 1.6400908e-01	[ 1.1410260e+00]	 2.0032144e-01


.. parsed-literal::

      63	 1.1812253e+00	 1.6058919e-01	 1.2307583e+00	 1.6147710e-01	[ 1.1507924e+00]	 2.0488143e-01
      64	 1.1906306e+00	 1.5860569e-01	 1.2401454e+00	 1.5962248e-01	[ 1.1626489e+00]	 1.8849635e-01


.. parsed-literal::

      65	 1.1970875e+00	 1.5758095e-01	 1.2465850e+00	 1.5893128e-01	[ 1.1647740e+00]	 1.8573141e-01


.. parsed-literal::

      66	 1.2089306e+00	 1.5516027e-01	 1.2588925e+00	 1.5742602e-01	[ 1.1678867e+00]	 2.2330904e-01


.. parsed-literal::

      67	 1.2166914e+00	 1.5296978e-01	 1.2669067e+00	 1.5566157e-01	[ 1.1724165e+00]	 2.0413923e-01


.. parsed-literal::

      68	 1.2247090e+00	 1.5245172e-01	 1.2748038e+00	 1.5514144e-01	[ 1.1840301e+00]	 2.1794391e-01


.. parsed-literal::

      69	 1.2302307e+00	 1.5209603e-01	 1.2804110e+00	 1.5452264e-01	[ 1.1935119e+00]	 2.1268892e-01


.. parsed-literal::

      70	 1.2348007e+00	 1.5215589e-01	 1.2850247e+00	 1.5428630e-01	[ 1.1994695e+00]	 2.2335267e-01


.. parsed-literal::

      71	 1.2459118e+00	 1.5204664e-01	 1.2963024e+00	 1.5353163e-01	[ 1.2117394e+00]	 2.1118450e-01


.. parsed-literal::

      72	 1.2469738e+00	 1.5439879e-01	 1.2978852e+00	 1.5472349e-01	  1.2082792e+00 	 2.1057963e-01


.. parsed-literal::

      73	 1.2565846e+00	 1.5256976e-01	 1.3070366e+00	 1.5345356e-01	[ 1.2194506e+00]	 2.1768665e-01


.. parsed-literal::

      74	 1.2604998e+00	 1.5147398e-01	 1.3109999e+00	 1.5256406e-01	[ 1.2213935e+00]	 2.2014403e-01


.. parsed-literal::

      75	 1.2677375e+00	 1.5015848e-01	 1.3184418e+00	 1.5133722e-01	[ 1.2250774e+00]	 2.1188045e-01


.. parsed-literal::

      76	 1.2773256e+00	 1.4875378e-01	 1.3284168e+00	 1.4979353e-01	[ 1.2315326e+00]	 2.1148372e-01


.. parsed-literal::

      77	 1.2821523e+00	 1.4790437e-01	 1.3335156e+00	 1.4900520e-01	[ 1.2346306e+00]	 3.2212543e-01


.. parsed-literal::

      78	 1.2872563e+00	 1.4785609e-01	 1.3387229e+00	 1.4873599e-01	[ 1.2400758e+00]	 2.2295189e-01


.. parsed-literal::

      79	 1.2915221e+00	 1.4781437e-01	 1.3429762e+00	 1.4864126e-01	[ 1.2454520e+00]	 2.1679688e-01
      80	 1.2981632e+00	 1.4760034e-01	 1.3497464e+00	 1.4843132e-01	[ 1.2492968e+00]	 1.7512870e-01


.. parsed-literal::

      81	 1.3040623e+00	 1.4712671e-01	 1.3558420e+00	 1.4832376e-01	  1.2481440e+00 	 2.0452952e-01
      82	 1.3108250e+00	 1.4698003e-01	 1.3626111e+00	 1.4821627e-01	[ 1.2506321e+00]	 1.9439483e-01


.. parsed-literal::

      83	 1.3146897e+00	 1.4626428e-01	 1.3663948e+00	 1.4769711e-01	[ 1.2523788e+00]	 2.1618319e-01


.. parsed-literal::

      84	 1.3193768e+00	 1.4539319e-01	 1.3713636e+00	 1.4705343e-01	  1.2502452e+00 	 2.1012878e-01
      85	 1.3238347e+00	 1.4471191e-01	 1.3761056e+00	 1.4671366e-01	  1.2496596e+00 	 1.7643857e-01


.. parsed-literal::

      86	 1.3282981e+00	 1.4409201e-01	 1.3807372e+00	 1.4642846e-01	  1.2496537e+00 	 2.1058488e-01
      87	 1.3319536e+00	 1.4375160e-01	 1.3844369e+00	 1.4615222e-01	  1.2507513e+00 	 1.7434049e-01


.. parsed-literal::

      88	 1.3375775e+00	 1.4286710e-01	 1.3901497e+00	 1.4534647e-01	[ 1.2572208e+00]	 2.1068811e-01


.. parsed-literal::

      89	 1.3437366e+00	 1.4186845e-01	 1.3965302e+00	 1.4467612e-01	[ 1.2579667e+00]	 2.1093392e-01
      90	 1.3482230e+00	 1.4109325e-01	 1.4010056e+00	 1.4413813e-01	[ 1.2636730e+00]	 1.7347860e-01


.. parsed-literal::

      91	 1.3519640e+00	 1.4071324e-01	 1.4048075e+00	 1.4405062e-01	[ 1.2675029e+00]	 2.0685101e-01
      92	 1.3559617e+00	 1.4015282e-01	 1.4090225e+00	 1.4412105e-01	  1.2657476e+00 	 2.0145655e-01


.. parsed-literal::

      93	 1.3605410e+00	 1.4012646e-01	 1.4135957e+00	 1.4446803e-01	[ 1.2712199e+00]	 2.1294689e-01


.. parsed-literal::

      94	 1.3655745e+00	 1.3984547e-01	 1.4187533e+00	 1.4475394e-01	  1.2705018e+00 	 2.0951247e-01


.. parsed-literal::

      95	 1.3692284e+00	 1.4000377e-01	 1.4223668e+00	 1.4511369e-01	[ 1.2735566e+00]	 2.1257353e-01


.. parsed-literal::

      96	 1.3725652e+00	 1.3953934e-01	 1.4257281e+00	 1.4478243e-01	  1.2716851e+00 	 2.1478820e-01
      97	 1.3775156e+00	 1.3865990e-01	 1.4308499e+00	 1.4412337e-01	  1.2667005e+00 	 2.0332313e-01


.. parsed-literal::

      98	 1.3806791e+00	 1.3816011e-01	 1.4342475e+00	 1.4345047e-01	  1.2606875e+00 	 2.0355654e-01
      99	 1.3845917e+00	 1.3774688e-01	 1.4381317e+00	 1.4300573e-01	  1.2630907e+00 	 2.0370674e-01


.. parsed-literal::

     100	 1.3871860e+00	 1.3754421e-01	 1.4407303e+00	 1.4271917e-01	  1.2664901e+00 	 2.1731949e-01


.. parsed-literal::

     101	 1.3900073e+00	 1.3726951e-01	 1.4436043e+00	 1.4218368e-01	  1.2665289e+00 	 2.0938993e-01


.. parsed-literal::

     102	 1.3919518e+00	 1.3668945e-01	 1.4457987e+00	 1.4129730e-01	  1.2627073e+00 	 2.0658541e-01
     103	 1.3961523e+00	 1.3648429e-01	 1.4498351e+00	 1.4099704e-01	  1.2659537e+00 	 2.1062660e-01


.. parsed-literal::

     104	 1.3981743e+00	 1.3617497e-01	 1.4518826e+00	 1.4060896e-01	  1.2655494e+00 	 2.1868086e-01


.. parsed-literal::

     105	 1.4007271e+00	 1.3574052e-01	 1.4544890e+00	 1.4006461e-01	  1.2663676e+00 	 2.1906042e-01


.. parsed-literal::

     106	 1.4051588e+00	 1.3487571e-01	 1.4589793e+00	 1.3901140e-01	  1.2634183e+00 	 2.1069527e-01


.. parsed-literal::

     107	 1.4078324e+00	 1.3440742e-01	 1.4617991e+00	 1.3846181e-01	  1.2660162e+00 	 3.1709385e-01


.. parsed-literal::

     108	 1.4104296e+00	 1.3396570e-01	 1.4643381e+00	 1.3796585e-01	  1.2635574e+00 	 2.1051478e-01


.. parsed-literal::

     109	 1.4130869e+00	 1.3360182e-01	 1.4669684e+00	 1.3765919e-01	  1.2615599e+00 	 2.2192860e-01


.. parsed-literal::

     110	 1.4165106e+00	 1.3322973e-01	 1.4704385e+00	 1.3720791e-01	  1.2593967e+00 	 2.1228075e-01


.. parsed-literal::

     111	 1.4185914e+00	 1.3258353e-01	 1.4728329e+00	 1.3655014e-01	  1.2436099e+00 	 2.2005081e-01


.. parsed-literal::

     112	 1.4222211e+00	 1.3265182e-01	 1.4763141e+00	 1.3646943e-01	  1.2514348e+00 	 2.1982861e-01


.. parsed-literal::

     113	 1.4241200e+00	 1.3260391e-01	 1.4782064e+00	 1.3625459e-01	  1.2542320e+00 	 2.2076201e-01
     114	 1.4268031e+00	 1.3236614e-01	 1.4809679e+00	 1.3581177e-01	  1.2533822e+00 	 1.9688821e-01


.. parsed-literal::

     115	 1.4307024e+00	 1.3193164e-01	 1.4849777e+00	 1.3513644e-01	  1.2507828e+00 	 2.1254563e-01


.. parsed-literal::

     116	 1.4328264e+00	 1.3126435e-01	 1.4873169e+00	 1.3424912e-01	  1.2380360e+00 	 2.1574283e-01


.. parsed-literal::

     117	 1.4357341e+00	 1.3129174e-01	 1.4901084e+00	 1.3440902e-01	  1.2427702e+00 	 2.1250367e-01
     118	 1.4374594e+00	 1.3118106e-01	 1.4918320e+00	 1.3434174e-01	  1.2402088e+00 	 2.0004177e-01


.. parsed-literal::

     119	 1.4401612e+00	 1.3088434e-01	 1.4946004e+00	 1.3402703e-01	  1.2337370e+00 	 2.0609164e-01
     120	 1.4409709e+00	 1.3038190e-01	 1.4957653e+00	 1.3335804e-01	  1.2056575e+00 	 2.0319986e-01


.. parsed-literal::

     121	 1.4450622e+00	 1.3012453e-01	 1.4996812e+00	 1.3301390e-01	  1.2133435e+00 	 2.0589709e-01


.. parsed-literal::

     122	 1.4465127e+00	 1.2998111e-01	 1.5011028e+00	 1.3276718e-01	  1.2166809e+00 	 2.1275663e-01
     123	 1.4486322e+00	 1.2962564e-01	 1.5032537e+00	 1.3218699e-01	  1.2143238e+00 	 1.7180204e-01


.. parsed-literal::

     124	 1.4504866e+00	 1.2917941e-01	 1.5052087e+00	 1.3152784e-01	  1.2220732e+00 	 1.8515921e-01
     125	 1.4529015e+00	 1.2886971e-01	 1.5075922e+00	 1.3101326e-01	  1.2127892e+00 	 1.9752979e-01


.. parsed-literal::

     126	 1.4548684e+00	 1.2866689e-01	 1.5095427e+00	 1.3068157e-01	  1.2058536e+00 	 2.0098615e-01
     127	 1.4566958e+00	 1.2845996e-01	 1.5113912e+00	 1.3030666e-01	  1.2025342e+00 	 1.9609022e-01


.. parsed-literal::

     128	 1.4586973e+00	 1.2827348e-01	 1.5135007e+00	 1.2991057e-01	  1.1905210e+00 	 2.0209146e-01


.. parsed-literal::

     129	 1.4612980e+00	 1.2808826e-01	 1.5161013e+00	 1.2952551e-01	  1.1938515e+00 	 2.0650339e-01
     130	 1.4629632e+00	 1.2802943e-01	 1.5177790e+00	 1.2942265e-01	  1.1970073e+00 	 1.7072439e-01


.. parsed-literal::

     131	 1.4649490e+00	 1.2793445e-01	 1.5198262e+00	 1.2930842e-01	  1.1946258e+00 	 2.0711279e-01


.. parsed-literal::

     132	 1.4670656e+00	 1.2759506e-01	 1.5220792e+00	 1.2897324e-01	  1.1868584e+00 	 2.1711183e-01
     133	 1.4694509e+00	 1.2743862e-01	 1.5245268e+00	 1.2895664e-01	  1.1734299e+00 	 2.0563722e-01


.. parsed-literal::

     134	 1.4710572e+00	 1.2732540e-01	 1.5261518e+00	 1.2897895e-01	  1.1662305e+00 	 2.1060491e-01
     135	 1.4730360e+00	 1.2703907e-01	 1.5282062e+00	 1.2884581e-01	  1.1556763e+00 	 1.8554616e-01


.. parsed-literal::

     136	 1.4741209e+00	 1.2684038e-01	 1.5294458e+00	 1.2892424e-01	  1.1493684e+00 	 2.1621180e-01


.. parsed-literal::

     137	 1.4758406e+00	 1.2679404e-01	 1.5311109e+00	 1.2879406e-01	  1.1499212e+00 	 2.0729065e-01


.. parsed-literal::

     138	 1.4770976e+00	 1.2671572e-01	 1.5324014e+00	 1.2864369e-01	  1.1493458e+00 	 2.1868753e-01


.. parsed-literal::

     139	 1.4780242e+00	 1.2663274e-01	 1.5333630e+00	 1.2848567e-01	  1.1477981e+00 	 2.0762873e-01
     140	 1.4798120e+00	 1.2643914e-01	 1.5352372e+00	 1.2816566e-01	  1.1435492e+00 	 1.8099952e-01


.. parsed-literal::

     141	 1.4815374e+00	 1.2623914e-01	 1.5370069e+00	 1.2753843e-01	  1.1293117e+00 	 2.1986127e-01


.. parsed-literal::

     142	 1.4830769e+00	 1.2618580e-01	 1.5385036e+00	 1.2747257e-01	  1.1302953e+00 	 2.1139240e-01
     143	 1.4847835e+00	 1.2608431e-01	 1.5401987e+00	 1.2733875e-01	  1.1272829e+00 	 1.9861317e-01


.. parsed-literal::

     144	 1.4863182e+00	 1.2598042e-01	 1.5417534e+00	 1.2705381e-01	  1.1194817e+00 	 2.1383595e-01


.. parsed-literal::

     145	 1.4886444e+00	 1.2575158e-01	 1.5442143e+00	 1.2648541e-01	  1.0997886e+00 	 2.1647859e-01


.. parsed-literal::

     146	 1.4903147e+00	 1.2565529e-01	 1.5458738e+00	 1.2611972e-01	  1.0866031e+00 	 2.1129990e-01


.. parsed-literal::

     147	 1.4914190e+00	 1.2560138e-01	 1.5469731e+00	 1.2600378e-01	  1.0803749e+00 	 2.1145630e-01


.. parsed-literal::

     148	 1.4934710e+00	 1.2547432e-01	 1.5490785e+00	 1.2568805e-01	  1.0596047e+00 	 2.1312404e-01


.. parsed-literal::

     149	 1.4940264e+00	 1.2531853e-01	 1.5497996e+00	 1.2539826e-01	  1.0256836e+00 	 2.0408154e-01


.. parsed-literal::

     150	 1.4961750e+00	 1.2527110e-01	 1.5518424e+00	 1.2536548e-01	  1.0342460e+00 	 2.1456504e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.07 s, total: 2min 7s
    Wall time: 32.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fdb0cb34b50>



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
    CPU times: user 2.16 s, sys: 40.9 ms, total: 2.2 s
    Wall time: 682 ms


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

