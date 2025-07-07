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
       1	-3.3890695e-01	 3.1903898e-01	-3.2913598e-01	 3.2768156e-01	[-3.4383588e-01]	 4.7337580e-01


.. parsed-literal::

       2	-2.6716136e-01	 3.0813856e-01	-2.4330183e-01	 3.1701591e-01	[-2.6679026e-01]	 2.3054099e-01


.. parsed-literal::

       3	-2.2192933e-01	 2.8788437e-01	-1.8085404e-01	 2.9704021e-01	[-2.1266162e-01]	 3.0226803e-01


.. parsed-literal::

       4	-1.8181464e-01	 2.6986026e-01	-1.3326796e-01	 2.7995048e-01	[-1.7567801e-01]	 2.9214811e-01


.. parsed-literal::

       5	-1.3170567e-01	 2.5651923e-01	-1.0160015e-01	 2.6833425e-01	[-1.5673972e-01]	 2.0830703e-01
       6	-6.9602510e-02	 2.5210471e-01	-4.1011292e-02	 2.6146373e-01	[-7.7041456e-02]	 1.9972110e-01


.. parsed-literal::

       7	-4.8743431e-02	 2.4791107e-01	-2.4957326e-02	 2.5929145e-01	[-6.4032924e-02]	 2.1224928e-01


.. parsed-literal::

       8	-3.7155140e-02	 2.4587154e-01	-1.6329170e-02	 2.5605412e-01	[-5.4963380e-02]	 2.1094871e-01


.. parsed-literal::

       9	-2.3385403e-02	 2.4332808e-01	-5.5512170e-03	 2.5274769e-01	[-4.0747728e-02]	 2.0920300e-01


.. parsed-literal::

      10	-1.2535411e-02	 2.4131241e-01	 3.1719003e-03	 2.5005048e-01	[-3.1781364e-02]	 2.2155476e-01
      11	-5.0580275e-03	 2.4003752e-01	 9.2512363e-03	 2.4890432e-01	[-2.4789820e-02]	 1.9213748e-01


.. parsed-literal::

      12	-1.2593745e-03	 2.3930910e-01	 1.2743750e-02	 2.4818849e-01	[-2.0725534e-02]	 2.1533084e-01


.. parsed-literal::

      13	 2.9709706e-03	 2.3855906e-01	 1.6871049e-02	 2.4728208e-01	[-1.6480165e-02]	 2.2221494e-01


.. parsed-literal::

      14	 8.4798645e-02	 2.2647003e-01	 1.0329078e-01	 2.3721822e-01	[ 7.1607777e-02]	 3.3363962e-01


.. parsed-literal::

      15	 1.1461135e-01	 2.2542321e-01	 1.3695628e-01	 2.3347696e-01	[ 1.1222210e-01]	 3.2567692e-01


.. parsed-literal::

      16	 1.5188386e-01	 2.1906214e-01	 1.7451985e-01	 2.2906400e-01	[ 1.5025629e-01]	 2.1654987e-01


.. parsed-literal::

      17	 2.4652908e-01	 2.1417535e-01	 2.7419175e-01	 2.2453884e-01	[ 2.4120086e-01]	 2.2327971e-01


.. parsed-literal::

      18	 3.1570646e-01	 2.0771587e-01	 3.4960121e-01	 2.2336464e-01	[ 2.9357716e-01]	 2.1267343e-01


.. parsed-literal::

      19	 3.5168344e-01	 2.0528588e-01	 3.8557012e-01	 2.2340720e-01	[ 3.2494241e-01]	 2.1172142e-01


.. parsed-literal::

      20	 3.9355225e-01	 2.0041380e-01	 4.2747380e-01	 2.1822759e-01	[ 3.6694562e-01]	 2.0726895e-01
      21	 4.3988933e-01	 1.9618701e-01	 4.7414908e-01	 2.1360805e-01	[ 4.1929287e-01]	 2.0114565e-01


.. parsed-literal::

      22	 5.2640446e-01	 1.9302826e-01	 5.6026784e-01	 2.0852534e-01	[ 5.1696553e-01]	 1.8158007e-01
      23	 6.2894082e-01	 1.9171316e-01	 6.6543014e-01	 2.0694218e-01	[ 6.0976103e-01]	 1.7422652e-01


.. parsed-literal::

      24	 6.5734390e-01	 1.9078618e-01	 6.9944272e-01	 2.0393055e-01	[ 6.4825880e-01]	 1.7997169e-01
      25	 7.0550677e-01	 1.8641698e-01	 7.4433855e-01	 2.0051861e-01	[ 6.9285465e-01]	 1.8819618e-01


.. parsed-literal::

      26	 7.2601630e-01	 1.8383188e-01	 7.6439584e-01	 1.9959911e-01	[ 7.1175111e-01]	 2.1216559e-01


.. parsed-literal::

      27	 7.4823727e-01	 1.8479809e-01	 7.8611282e-01	 2.0043705e-01	[ 7.3690706e-01]	 2.1380472e-01


.. parsed-literal::

      28	 7.7915297e-01	 1.8255483e-01	 8.1814991e-01	 1.9900941e-01	[ 7.7187407e-01]	 2.1320701e-01
      29	 8.0147972e-01	 1.8270406e-01	 8.4161414e-01	 2.0019223e-01	[ 7.9184460e-01]	 2.0516300e-01


.. parsed-literal::

      30	 8.2634392e-01	 1.8728661e-01	 8.6718828e-01	 2.0454575e-01	[ 8.1824026e-01]	 2.0329571e-01


.. parsed-literal::

      31	 8.5022108e-01	 1.9025062e-01	 8.9086019e-01	 2.0514447e-01	[ 8.4202184e-01]	 2.1300030e-01
      32	 8.6879845e-01	 1.9001160e-01	 9.0932225e-01	 2.0422007e-01	[ 8.6253722e-01]	 1.9865894e-01


.. parsed-literal::

      33	 8.9850942e-01	 1.9200293e-01	 9.3948284e-01	 2.0578164e-01	[ 8.9084371e-01]	 1.9740367e-01


.. parsed-literal::

      34	 9.0727063e-01	 1.9702284e-01	 9.5043541e-01	 2.0945379e-01	[ 9.0794008e-01]	 2.1228886e-01
      35	 9.3509606e-01	 1.9146165e-01	 9.7754079e-01	 2.0487808e-01	[ 9.2784237e-01]	 2.0604587e-01


.. parsed-literal::

      36	 9.5295342e-01	 1.8658446e-01	 9.9526870e-01	 2.0081668e-01	[ 9.4125104e-01]	 2.1265888e-01


.. parsed-literal::

      37	 9.7504180e-01	 1.7999934e-01	 1.0179311e+00	 1.9580477e-01	[ 9.6319949e-01]	 2.0800972e-01


.. parsed-literal::

      38	 1.0038606e+00	 1.7146669e-01	 1.0478951e+00	 1.8776591e-01	[ 9.9231753e-01]	 2.1810937e-01


.. parsed-literal::

      39	 1.0161136e+00	 1.6771128e-01	 1.0612828e+00	 1.8447091e-01	[ 1.0084795e+00]	 2.0786953e-01
      40	 1.0288322e+00	 1.6615923e-01	 1.0737497e+00	 1.8295060e-01	[ 1.0235585e+00]	 1.7702198e-01


.. parsed-literal::

      41	 1.0475386e+00	 1.6269756e-01	 1.0927254e+00	 1.8009849e-01	[ 1.0442167e+00]	 2.1239114e-01


.. parsed-literal::

      42	 1.0683664e+00	 1.5885581e-01	 1.1140190e+00	 1.7561719e-01	[ 1.0644573e+00]	 2.1987891e-01


.. parsed-literal::

      43	 1.0756532e+00	 1.5464689e-01	 1.1228391e+00	 1.7028026e-01	[ 1.0708108e+00]	 2.0838332e-01


.. parsed-literal::

      44	 1.0959408e+00	 1.5467040e-01	 1.1425399e+00	 1.7008619e-01	[ 1.0846417e+00]	 2.1029925e-01


.. parsed-literal::

      45	 1.1038678e+00	 1.5358974e-01	 1.1505386e+00	 1.6926026e-01	[ 1.0899549e+00]	 2.1214724e-01


.. parsed-literal::

      46	 1.1161467e+00	 1.5210372e-01	 1.1630784e+00	 1.6759652e-01	[ 1.0973769e+00]	 2.1109343e-01


.. parsed-literal::

      47	 1.1270011e+00	 1.4894468e-01	 1.1746233e+00	 1.6407449e-01	[ 1.1046243e+00]	 2.1107292e-01
      48	 1.1421924e+00	 1.4744490e-01	 1.1896438e+00	 1.6286415e-01	[ 1.1187887e+00]	 1.9866347e-01


.. parsed-literal::

      49	 1.1521733e+00	 1.4621964e-01	 1.1997352e+00	 1.6199753e-01	[ 1.1306516e+00]	 2.0807719e-01


.. parsed-literal::

      50	 1.1657829e+00	 1.4419070e-01	 1.2138011e+00	 1.6035092e-01	[ 1.1458590e+00]	 2.1997666e-01


.. parsed-literal::

      51	 1.1816525e+00	 1.4082950e-01	 1.2299361e+00	 1.5807536e-01	[ 1.1647407e+00]	 2.1426964e-01


.. parsed-literal::

      52	 1.1963905e+00	 1.3896067e-01	 1.2452141e+00	 1.5719199e-01	[ 1.1757300e+00]	 2.2043586e-01


.. parsed-literal::

      53	 1.2069803e+00	 1.3827164e-01	 1.2558351e+00	 1.5659902e-01	[ 1.1834257e+00]	 2.1015692e-01
      54	 1.2216876e+00	 1.3670387e-01	 1.2708132e+00	 1.5597221e-01	[ 1.1919454e+00]	 1.9021583e-01


.. parsed-literal::

      55	 1.2313400e+00	 1.3543944e-01	 1.2806253e+00	 1.5558201e-01	[ 1.1957809e+00]	 2.0925045e-01


.. parsed-literal::

      56	 1.2425254e+00	 1.3511745e-01	 1.2919323e+00	 1.5543420e-01	[ 1.2002013e+00]	 2.0627999e-01


.. parsed-literal::

      57	 1.2527312e+00	 1.3394032e-01	 1.3024523e+00	 1.5453675e-01	[ 1.2090500e+00]	 2.0357704e-01


.. parsed-literal::

      58	 1.2633676e+00	 1.3303600e-01	 1.3138155e+00	 1.5359172e-01	[ 1.2168091e+00]	 2.1192098e-01


.. parsed-literal::

      59	 1.2753591e+00	 1.3226861e-01	 1.3259307e+00	 1.5268435e-01	[ 1.2275530e+00]	 2.1231246e-01


.. parsed-literal::

      60	 1.2851809e+00	 1.3176589e-01	 1.3365124e+00	 1.5281614e-01	  1.2267351e+00 	 2.1143198e-01


.. parsed-literal::

      61	 1.2942203e+00	 1.3110816e-01	 1.3454378e+00	 1.5248525e-01	[ 1.2329934e+00]	 2.0883322e-01


.. parsed-literal::

      62	 1.3050245e+00	 1.3001097e-01	 1.3563116e+00	 1.5271091e-01	[ 1.2353235e+00]	 2.1859598e-01


.. parsed-literal::

      63	 1.3150268e+00	 1.2885359e-01	 1.3661434e+00	 1.5328482e-01	[ 1.2394531e+00]	 2.1100855e-01


.. parsed-literal::

      64	 1.3221849e+00	 1.2821677e-01	 1.3736112e+00	 1.5472010e-01	  1.2316237e+00 	 2.1092963e-01


.. parsed-literal::

      65	 1.3294380e+00	 1.2780122e-01	 1.3805641e+00	 1.5448800e-01	  1.2378893e+00 	 2.1197844e-01
      66	 1.3365658e+00	 1.2734562e-01	 1.3877953e+00	 1.5464310e-01	  1.2362519e+00 	 1.8012762e-01


.. parsed-literal::

      67	 1.3435316e+00	 1.2688741e-01	 1.3947626e+00	 1.5498090e-01	  1.2358449e+00 	 2.1367216e-01


.. parsed-literal::

      68	 1.3513074e+00	 1.2589168e-01	 1.4035304e+00	 1.5575532e-01	  1.2089161e+00 	 2.1956682e-01
      69	 1.3606029e+00	 1.2536813e-01	 1.4124067e+00	 1.5506844e-01	  1.2295570e+00 	 1.8627882e-01


.. parsed-literal::

      70	 1.3651668e+00	 1.2512891e-01	 1.4168937e+00	 1.5473872e-01	  1.2370912e+00 	 1.8323803e-01
      71	 1.3721486e+00	 1.2437185e-01	 1.4245065e+00	 1.5472386e-01	  1.2360775e+00 	 1.9602609e-01


.. parsed-literal::

      72	 1.3777601e+00	 1.2401459e-01	 1.4302640e+00	 1.5453584e-01	  1.2289562e+00 	 2.0770359e-01


.. parsed-literal::

      73	 1.3830667e+00	 1.2387026e-01	 1.4354523e+00	 1.5484892e-01	  1.2311738e+00 	 2.1232891e-01
      74	 1.3902674e+00	 1.2348052e-01	 1.4427812e+00	 1.5505212e-01	  1.2311101e+00 	 1.8514872e-01


.. parsed-literal::

      75	 1.3951796e+00	 1.2332111e-01	 1.4478998e+00	 1.5546413e-01	  1.2299766e+00 	 2.1120024e-01
      76	 1.4006878e+00	 1.2305088e-01	 1.4534404e+00	 1.5537467e-01	  1.2353657e+00 	 1.8895435e-01


.. parsed-literal::

      77	 1.4051986e+00	 1.2243964e-01	 1.4582283e+00	 1.5445728e-01	[ 1.2405657e+00]	 2.0924640e-01
      78	 1.4091152e+00	 1.2242740e-01	 1.4621813e+00	 1.5461508e-01	  1.2375042e+00 	 1.8381357e-01


.. parsed-literal::

      79	 1.4115290e+00	 1.2226936e-01	 1.4642997e+00	 1.5429030e-01	[ 1.2443252e+00]	 2.0793915e-01


.. parsed-literal::

      80	 1.4147074e+00	 1.2208799e-01	 1.4674193e+00	 1.5413259e-01	[ 1.2478400e+00]	 2.0400143e-01


.. parsed-literal::

      81	 1.4195098e+00	 1.2171496e-01	 1.4722187e+00	 1.5367815e-01	[ 1.2534618e+00]	 2.1193004e-01


.. parsed-literal::

      82	 1.4234706e+00	 1.2167655e-01	 1.4762329e+00	 1.5316925e-01	[ 1.2552118e+00]	 2.0163393e-01
      83	 1.4284767e+00	 1.2131930e-01	 1.4812015e+00	 1.5261212e-01	[ 1.2630253e+00]	 1.7855215e-01


.. parsed-literal::

      84	 1.4319133e+00	 1.2108503e-01	 1.4847182e+00	 1.5207785e-01	[ 1.2678989e+00]	 2.1588135e-01
      85	 1.4352565e+00	 1.2091380e-01	 1.4881866e+00	 1.5169437e-01	[ 1.2703631e+00]	 1.9531083e-01


.. parsed-literal::

      86	 1.4407402e+00	 1.2040252e-01	 1.4939146e+00	 1.5103945e-01	[ 1.2743751e+00]	 2.1304846e-01


.. parsed-literal::

      87	 1.4444104e+00	 1.2019469e-01	 1.4977056e+00	 1.5072054e-01	[ 1.2793219e+00]	 2.1480656e-01


.. parsed-literal::

      88	 1.4470931e+00	 1.2003560e-01	 1.5002410e+00	 1.5076757e-01	[ 1.2816656e+00]	 2.0178294e-01


.. parsed-literal::

      89	 1.4497542e+00	 1.1982668e-01	 1.5028348e+00	 1.5064209e-01	[ 1.2847648e+00]	 2.0924520e-01
      90	 1.4529239e+00	 1.1958732e-01	 1.5060107e+00	 1.5050655e-01	[ 1.2859537e+00]	 2.0121050e-01


.. parsed-literal::

      91	 1.4564681e+00	 1.1932857e-01	 1.5096264e+00	 1.5011937e-01	[ 1.2877178e+00]	 2.0558548e-01


.. parsed-literal::

      92	 1.4595434e+00	 1.1930204e-01	 1.5127310e+00	 1.4991608e-01	[ 1.2878704e+00]	 2.0635223e-01
      93	 1.4628433e+00	 1.1924559e-01	 1.5161588e+00	 1.4958720e-01	[ 1.2896175e+00]	 1.7197895e-01


.. parsed-literal::

      94	 1.4660716e+00	 1.1915524e-01	 1.5194787e+00	 1.4933227e-01	  1.2841819e+00 	 2.1649098e-01


.. parsed-literal::

      95	 1.4693033e+00	 1.1891333e-01	 1.5227175e+00	 1.4920118e-01	  1.2849520e+00 	 2.1505308e-01


.. parsed-literal::

      96	 1.4727759e+00	 1.1847163e-01	 1.5261782e+00	 1.4901628e-01	  1.2854975e+00 	 2.1116996e-01
      97	 1.4755152e+00	 1.1826544e-01	 1.5288936e+00	 1.4904618e-01	  1.2868828e+00 	 1.7485237e-01


.. parsed-literal::

      98	 1.4799539e+00	 1.1795119e-01	 1.5332745e+00	 1.4913540e-01	  1.2877247e+00 	 2.1117520e-01
      99	 1.4840191e+00	 1.1759958e-01	 1.5374757e+00	 1.4891421e-01	  1.2838472e+00 	 1.8531656e-01


.. parsed-literal::

     100	 1.4875183e+00	 1.1750324e-01	 1.5410442e+00	 1.4894963e-01	  1.2861285e+00 	 2.0549226e-01


.. parsed-literal::

     101	 1.4896234e+00	 1.1748997e-01	 1.5431589e+00	 1.4872249e-01	  1.2867935e+00 	 2.1361136e-01


.. parsed-literal::

     102	 1.4918511e+00	 1.1752596e-01	 1.5454348e+00	 1.4853856e-01	  1.2875066e+00 	 2.1293926e-01
     103	 1.4935931e+00	 1.1746728e-01	 1.5472461e+00	 1.4815042e-01	[ 1.2902698e+00]	 1.7633939e-01


.. parsed-literal::

     104	 1.4961488e+00	 1.1740649e-01	 1.5497267e+00	 1.4813982e-01	[ 1.2923687e+00]	 1.9233465e-01
     105	 1.4980638e+00	 1.1726197e-01	 1.5515535e+00	 1.4802791e-01	[ 1.2957675e+00]	 1.9488430e-01


.. parsed-literal::

     106	 1.4999780e+00	 1.1706113e-01	 1.5534459e+00	 1.4778403e-01	[ 1.2969153e+00]	 2.1745348e-01
     107	 1.5031059e+00	 1.1662752e-01	 1.5567422e+00	 1.4720148e-01	[ 1.3014277e+00]	 1.9090629e-01


.. parsed-literal::

     108	 1.5064829e+00	 1.1629493e-01	 1.5602087e+00	 1.4667962e-01	  1.2974855e+00 	 2.0981383e-01


.. parsed-literal::

     109	 1.5086573e+00	 1.1624358e-01	 1.5624555e+00	 1.4656615e-01	  1.2977427e+00 	 2.1711421e-01
     110	 1.5104191e+00	 1.1617851e-01	 1.5643616e+00	 1.4638214e-01	  1.2996518e+00 	 1.9960809e-01


.. parsed-literal::

     111	 1.5126686e+00	 1.1605873e-01	 1.5667245e+00	 1.4626844e-01	  1.2989746e+00 	 2.0325971e-01


.. parsed-literal::

     112	 1.5154850e+00	 1.1595043e-01	 1.5696246e+00	 1.4625441e-01	[ 1.3038189e+00]	 2.1046996e-01


.. parsed-literal::

     113	 1.5176840e+00	 1.1580944e-01	 1.5718211e+00	 1.4628110e-01	  1.3024856e+00 	 2.1172905e-01


.. parsed-literal::

     114	 1.5196494e+00	 1.1574931e-01	 1.5738439e+00	 1.4640610e-01	  1.3010534e+00 	 2.1964216e-01


.. parsed-literal::

     115	 1.5214886e+00	 1.1570133e-01	 1.5757628e+00	 1.4645774e-01	  1.2998598e+00 	 2.1919656e-01
     116	 1.5231904e+00	 1.1565866e-01	 1.5775459e+00	 1.4641236e-01	  1.2998962e+00 	 1.9412756e-01


.. parsed-literal::

     117	 1.5263141e+00	 1.1556853e-01	 1.5807840e+00	 1.4615086e-01	  1.2985479e+00 	 2.0776296e-01


.. parsed-literal::

     118	 1.5280514e+00	 1.1574507e-01	 1.5825862e+00	 1.4637068e-01	  1.2952309e+00 	 2.1255064e-01
     119	 1.5303790e+00	 1.1556820e-01	 1.5847671e+00	 1.4613169e-01	  1.2990694e+00 	 1.8303776e-01


.. parsed-literal::

     120	 1.5318928e+00	 1.1546309e-01	 1.5861907e+00	 1.4603014e-01	  1.3009086e+00 	 2.0827270e-01
     121	 1.5338618e+00	 1.1532840e-01	 1.5881090e+00	 1.4597175e-01	  1.3022781e+00 	 1.9959211e-01


.. parsed-literal::

     122	 1.5357754e+00	 1.1514202e-01	 1.5900562e+00	 1.4572680e-01	  1.3002388e+00 	 1.8511510e-01


.. parsed-literal::

     123	 1.5380783e+00	 1.1504689e-01	 1.5923420e+00	 1.4573732e-01	  1.3013529e+00 	 2.1302581e-01
     124	 1.5395279e+00	 1.1493577e-01	 1.5938621e+00	 1.4562425e-01	  1.3005456e+00 	 1.9846797e-01


.. parsed-literal::

     125	 1.5411755e+00	 1.1476729e-01	 1.5955911e+00	 1.4536895e-01	  1.2989958e+00 	 2.0193386e-01


.. parsed-literal::

     126	 1.5436352e+00	 1.1444110e-01	 1.5982551e+00	 1.4489908e-01	  1.3016430e+00 	 2.0721698e-01


.. parsed-literal::

     127	 1.5458211e+00	 1.1416617e-01	 1.6005089e+00	 1.4454091e-01	  1.2942020e+00 	 2.1103406e-01


.. parsed-literal::

     128	 1.5470931e+00	 1.1415697e-01	 1.6016779e+00	 1.4449166e-01	  1.2980918e+00 	 2.1504259e-01


.. parsed-literal::

     129	 1.5487005e+00	 1.1405794e-01	 1.6032087e+00	 1.4444077e-01	  1.3015909e+00 	 2.1233726e-01
     130	 1.5503031e+00	 1.1380901e-01	 1.6048675e+00	 1.4423665e-01	  1.2983050e+00 	 2.0238566e-01


.. parsed-literal::

     131	 1.5521414e+00	 1.1363050e-01	 1.6067543e+00	 1.4414041e-01	  1.2961710e+00 	 1.8529916e-01
     132	 1.5539347e+00	 1.1339407e-01	 1.6086516e+00	 1.4404219e-01	  1.2894632e+00 	 1.7971039e-01


.. parsed-literal::

     133	 1.5554813e+00	 1.1318921e-01	 1.6103318e+00	 1.4387205e-01	  1.2810088e+00 	 2.0860767e-01


.. parsed-literal::

     134	 1.5572474e+00	 1.1304267e-01	 1.6121845e+00	 1.4371925e-01	  1.2715319e+00 	 2.1006870e-01


.. parsed-literal::

     135	 1.5588346e+00	 1.1288040e-01	 1.6138342e+00	 1.4352109e-01	  1.2670924e+00 	 2.1603107e-01


.. parsed-literal::

     136	 1.5603768e+00	 1.1278530e-01	 1.6154098e+00	 1.4351834e-01	  1.2621230e+00 	 2.2216034e-01


.. parsed-literal::

     137	 1.5615110e+00	 1.1256794e-01	 1.6165885e+00	 1.4341678e-01	  1.2610722e+00 	 2.1556377e-01
     138	 1.5626414e+00	 1.1249676e-01	 1.6176627e+00	 1.4347101e-01	  1.2622061e+00 	 1.9862223e-01


.. parsed-literal::

     139	 1.5638630e+00	 1.1238687e-01	 1.6188328e+00	 1.4356415e-01	  1.2629060e+00 	 2.1165466e-01


.. parsed-literal::

     140	 1.5649783e+00	 1.1226359e-01	 1.6199334e+00	 1.4360661e-01	  1.2597155e+00 	 2.1959186e-01


.. parsed-literal::

     141	 1.5660516e+00	 1.1203654e-01	 1.6210697e+00	 1.4378960e-01	  1.2539863e+00 	 2.1350574e-01


.. parsed-literal::

     142	 1.5674510e+00	 1.1199544e-01	 1.6224458e+00	 1.4372345e-01	  1.2506574e+00 	 2.0989919e-01


.. parsed-literal::

     143	 1.5681054e+00	 1.1199454e-01	 1.6231042e+00	 1.4365570e-01	  1.2500003e+00 	 2.2531867e-01
     144	 1.5696060e+00	 1.1190601e-01	 1.6247038e+00	 1.4359943e-01	  1.2432153e+00 	 1.8845701e-01


.. parsed-literal::

     145	 1.5703910e+00	 1.1183010e-01	 1.6257778e+00	 1.4347629e-01	  1.2310705e+00 	 2.1265054e-01


.. parsed-literal::

     146	 1.5720283e+00	 1.1177252e-01	 1.6273727e+00	 1.4357985e-01	  1.2298909e+00 	 2.1495199e-01


.. parsed-literal::

     147	 1.5727971e+00	 1.1171874e-01	 1.6281380e+00	 1.4361528e-01	  1.2284612e+00 	 2.1017241e-01
     148	 1.5739786e+00	 1.1167948e-01	 1.6293667e+00	 1.4365142e-01	  1.2255360e+00 	 1.8722010e-01


.. parsed-literal::

     149	 1.5744802e+00	 1.1167249e-01	 1.6300130e+00	 1.4370802e-01	  1.2205774e+00 	 1.9374895e-01


.. parsed-literal::

     150	 1.5757507e+00	 1.1167265e-01	 1.6311923e+00	 1.4367457e-01	  1.2219371e+00 	 2.0923758e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.13 s, total: 2min 6s
    Wall time: 31.8 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f72f85ea800>



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
    CPU times: user 1.88 s, sys: 48 ms, total: 1.93 s
    Wall time: 658 ms


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

