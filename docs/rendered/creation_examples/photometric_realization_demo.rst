Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fe0c024b4c0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.890625</td>
          <td>26.746787</td>
          <td>0.455292</td>
          <td>26.291787</td>
          <td>0.115643</td>
          <td>26.075967</td>
          <td>0.084311</td>
          <td>25.437437</td>
          <td>0.078222</td>
          <td>24.970295</td>
          <td>0.098839</td>
          <td>25.052773</td>
          <td>0.232894</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.343571</td>
          <td>1.284736</td>
          <td>28.627590</td>
          <td>0.732318</td>
          <td>27.927922</td>
          <td>0.398160</td>
          <td>27.672908</td>
          <td>0.498800</td>
          <td>27.326106</td>
          <td>0.657344</td>
          <td>25.853789</td>
          <td>0.440254</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.937775</td>
          <td>0.240238</td>
          <td>25.946459</td>
          <td>0.085483</td>
          <td>24.797378</td>
          <td>0.027216</td>
          <td>23.892612</td>
          <td>0.020105</td>
          <td>23.140843</td>
          <td>0.019797</td>
          <td>22.803550</td>
          <td>0.032786</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.554411</td>
          <td>0.697045</td>
          <td>30.013544</td>
          <td>1.511424</td>
          <td>26.684815</td>
          <td>0.229046</td>
          <td>26.160746</td>
          <td>0.271930</td>
          <td>25.277456</td>
          <td>0.279993</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.237133</td>
          <td>0.306403</td>
          <td>25.731694</td>
          <td>0.070737</td>
          <td>25.462902</td>
          <td>0.048982</td>
          <td>24.896948</td>
          <td>0.048444</td>
          <td>24.296145</td>
          <td>0.054483</td>
          <td>23.633146</td>
          <td>0.068384</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>2.147172</td>
          <td>25.874456</td>
          <td>0.227996</td>
          <td>26.605420</td>
          <td>0.151625</td>
          <td>26.136465</td>
          <td>0.088924</td>
          <td>25.939733</td>
          <td>0.121496</td>
          <td>25.796091</td>
          <td>0.201116</td>
          <td>26.217743</td>
          <td>0.575532</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.702637</td>
          <td>0.883101</td>
          <td>27.140087</td>
          <td>0.237899</td>
          <td>27.279229</td>
          <td>0.236921</td>
          <td>26.241016</td>
          <td>0.157545</td>
          <td>26.031832</td>
          <td>0.244686</td>
          <td>25.349353</td>
          <td>0.296744</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.665095</td>
          <td>0.862375</td>
          <td>27.365296</td>
          <td>0.285986</td>
          <td>26.693843</td>
          <td>0.144499</td>
          <td>26.566773</td>
          <td>0.207587</td>
          <td>25.814227</td>
          <td>0.204199</td>
          <td>25.309331</td>
          <td>0.287314</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>34.405999</td>
          <td>6.951629</td>
          <td>27.559228</td>
          <td>0.334016</td>
          <td>26.392565</td>
          <td>0.111295</td>
          <td>25.989520</td>
          <td>0.126860</td>
          <td>25.840285</td>
          <td>0.208705</td>
          <td>25.199505</td>
          <td>0.262774</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.225174</td>
          <td>0.643603</td>
          <td>26.506633</td>
          <td>0.139285</td>
          <td>26.241333</td>
          <td>0.097505</td>
          <td>25.718162</td>
          <td>0.100137</td>
          <td>25.331735</td>
          <td>0.135383</td>
          <td>24.649899</td>
          <td>0.165978</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.890625</td>
          <td>28.728444</td>
          <td>1.668720</td>
          <td>26.527316</td>
          <td>0.163009</td>
          <td>26.160494</td>
          <td>0.106722</td>
          <td>25.523224</td>
          <td>0.099867</td>
          <td>25.007168</td>
          <td>0.119838</td>
          <td>24.761375</td>
          <td>0.214060</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.264358</td>
          <td>1.196663</td>
          <td>27.888541</td>
          <td>0.444601</td>
          <td>26.948965</td>
          <td>0.331686</td>
          <td>26.377048</td>
          <td>0.374056</td>
          <td>26.319793</td>
          <td>0.705323</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.895155</td>
          <td>0.568002</td>
          <td>25.920092</td>
          <td>0.098341</td>
          <td>24.771051</td>
          <td>0.031985</td>
          <td>23.888654</td>
          <td>0.024189</td>
          <td>23.131991</td>
          <td>0.023529</td>
          <td>22.882741</td>
          <td>0.042604</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.662848</td>
          <td>0.871457</td>
          <td>27.998566</td>
          <td>0.510739</td>
          <td>27.080602</td>
          <td>0.391064</td>
          <td>26.270461</td>
          <td>0.365259</td>
          <td>25.672714</td>
          <td>0.470067</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.238431</td>
          <td>0.341160</td>
          <td>25.822266</td>
          <td>0.088482</td>
          <td>25.324091</td>
          <td>0.051023</td>
          <td>24.906508</td>
          <td>0.057966</td>
          <td>24.451911</td>
          <td>0.073653</td>
          <td>23.616459</td>
          <td>0.079756</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>2.147172</td>
          <td>26.068767</td>
          <td>0.302384</td>
          <td>27.035339</td>
          <td>0.253981</td>
          <td>26.146505</td>
          <td>0.107651</td>
          <td>26.366412</td>
          <td>0.210383</td>
          <td>25.946907</td>
          <td>0.270524</td>
          <td>26.986059</td>
          <td>1.089544</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>29.775419</td>
          <td>2.563816</td>
          <td>26.972210</td>
          <td>0.237717</td>
          <td>26.747399</td>
          <td>0.177715</td>
          <td>26.781779</td>
          <td>0.291241</td>
          <td>26.017383</td>
          <td>0.282047</td>
          <td>25.236596</td>
          <td>0.316913</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.444222</td>
          <td>1.460430</td>
          <td>27.592641</td>
          <td>0.393371</td>
          <td>27.182280</td>
          <td>0.257534</td>
          <td>26.229531</td>
          <td>0.186001</td>
          <td>25.560219</td>
          <td>0.194847</td>
          <td>25.086333</td>
          <td>0.283129</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.157341</td>
          <td>0.282992</td>
          <td>26.414189</td>
          <td>0.137223</td>
          <td>25.994005</td>
          <td>0.155094</td>
          <td>25.926435</td>
          <td>0.268629</td>
          <td>26.335923</td>
          <td>0.730752</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.934127</td>
          <td>0.579590</td>
          <td>26.460366</td>
          <td>0.155313</td>
          <td>26.146019</td>
          <td>0.106439</td>
          <td>25.566751</td>
          <td>0.104824</td>
          <td>25.035022</td>
          <td>0.123995</td>
          <td>25.053450</td>
          <td>0.274958</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.890625</td>
          <td>27.447846</td>
          <td>0.748829</td>
          <td>27.118543</td>
          <td>0.233725</td>
          <td>25.986367</td>
          <td>0.077913</td>
          <td>25.271109</td>
          <td>0.067530</td>
          <td>25.060508</td>
          <td>0.106972</td>
          <td>25.100703</td>
          <td>0.242333</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.755638</td>
          <td>0.913330</td>
          <td>28.078651</td>
          <td>0.497662</td>
          <td>27.228711</td>
          <td>0.227415</td>
          <td>28.213501</td>
          <td>0.730800</td>
          <td>26.974310</td>
          <td>0.511948</td>
          <td>25.200572</td>
          <td>0.263245</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.975524</td>
          <td>0.565592</td>
          <td>25.808375</td>
          <td>0.081373</td>
          <td>24.720622</td>
          <td>0.027623</td>
          <td>23.884622</td>
          <td>0.021711</td>
          <td>23.182103</td>
          <td>0.022213</td>
          <td>22.843843</td>
          <td>0.037007</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.612932</td>
          <td>0.415980</td>
          <td>26.914205</td>
          <td>0.216444</td>
          <td>26.757000</td>
          <td>0.301992</td>
          <td>26.199915</td>
          <td>0.344480</td>
          <td>25.222043</td>
          <td>0.331029</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.224456</td>
          <td>0.303578</td>
          <td>25.885623</td>
          <td>0.081126</td>
          <td>25.429589</td>
          <td>0.047623</td>
          <td>24.883303</td>
          <td>0.047933</td>
          <td>24.410411</td>
          <td>0.060383</td>
          <td>23.754762</td>
          <td>0.076263</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>2.147172</td>
          <td>26.281848</td>
          <td>0.333994</td>
          <td>26.233651</td>
          <td>0.117719</td>
          <td>26.139368</td>
          <td>0.096459</td>
          <td>26.103549</td>
          <td>0.151753</td>
          <td>25.889074</td>
          <td>0.234153</td>
          <td>27.485968</td>
          <td>1.343923</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.196166</td>
          <td>0.636505</td>
          <td>26.736672</td>
          <td>0.171974</td>
          <td>26.752438</td>
          <td>0.154420</td>
          <td>26.364117</td>
          <td>0.177921</td>
          <td>26.519257</td>
          <td>0.367445</td>
          <td>25.602290</td>
          <td>0.368314</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>29.721437</td>
          <td>2.427284</td>
          <td>27.403303</td>
          <td>0.306639</td>
          <td>26.823023</td>
          <td>0.169268</td>
          <td>26.223261</td>
          <td>0.163102</td>
          <td>25.949537</td>
          <td>0.239356</td>
          <td>26.097797</td>
          <td>0.550360</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>31.193724</td>
          <td>3.862380</td>
          <td>27.270805</td>
          <td>0.290712</td>
          <td>26.584177</td>
          <td>0.147113</td>
          <td>25.805603</td>
          <td>0.121733</td>
          <td>25.687981</td>
          <td>0.204901</td>
          <td>25.619310</td>
          <td>0.408102</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.796223</td>
          <td>0.483492</td>
          <td>26.700885</td>
          <td>0.170014</td>
          <td>26.070765</td>
          <td>0.087268</td>
          <td>25.661904</td>
          <td>0.099281</td>
          <td>25.352794</td>
          <td>0.143225</td>
          <td>25.507719</td>
          <td>0.349170</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
