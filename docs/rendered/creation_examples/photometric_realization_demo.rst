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

    <pzflow.flow.Flow at 0x7f8347da82b0>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.752687</td>
          <td>0.171931</td>
          <td>26.130013</td>
          <td>0.088420</td>
          <td>25.379195</td>
          <td>0.074298</td>
          <td>24.999037</td>
          <td>0.101360</td>
          <td>24.915675</td>
          <td>0.207771</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>30.144722</td>
          <td>2.776237</td>
          <td>27.732745</td>
          <td>0.382685</td>
          <td>28.282759</td>
          <td>0.519854</td>
          <td>26.523861</td>
          <td>0.200250</td>
          <td>26.735078</td>
          <td>0.427754</td>
          <td>25.420819</td>
          <td>0.314253</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.565175</td>
          <td>0.396525</td>
          <td>25.899305</td>
          <td>0.082008</td>
          <td>24.821853</td>
          <td>0.027804</td>
          <td>23.864176</td>
          <td>0.019626</td>
          <td>23.143115</td>
          <td>0.019835</td>
          <td>22.829516</td>
          <td>0.033545</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.443117</td>
          <td>0.746426</td>
          <td>27.606694</td>
          <td>0.346777</td>
          <td>27.799881</td>
          <td>0.360454</td>
          <td>26.858309</td>
          <td>0.264209</td>
          <td>26.066546</td>
          <td>0.251773</td>
          <td>25.240713</td>
          <td>0.271757</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.960407</td>
          <td>0.244755</td>
          <td>25.807454</td>
          <td>0.075630</td>
          <td>25.376719</td>
          <td>0.045374</td>
          <td>24.815238</td>
          <td>0.045054</td>
          <td>24.278137</td>
          <td>0.053619</td>
          <td>23.559558</td>
          <td>0.064068</td>
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
          <td>26.873522</td>
          <td>0.500329</td>
          <td>26.252868</td>
          <td>0.111791</td>
          <td>26.261299</td>
          <td>0.099226</td>
          <td>26.162909</td>
          <td>0.147339</td>
          <td>26.000652</td>
          <td>0.238474</td>
          <td>26.847131</td>
          <td>0.879498</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.248488</td>
          <td>0.654083</td>
          <td>26.634920</td>
          <td>0.155504</td>
          <td>26.649627</td>
          <td>0.139099</td>
          <td>26.436403</td>
          <td>0.186025</td>
          <td>26.090337</td>
          <td>0.256734</td>
          <td>25.406661</td>
          <td>0.310715</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.453128</td>
          <td>0.306941</td>
          <td>26.688553</td>
          <td>0.143842</td>
          <td>26.629835</td>
          <td>0.218813</td>
          <td>26.339854</td>
          <td>0.314199</td>
          <td>25.353688</td>
          <td>0.297781</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.199073</td>
          <td>0.249743</td>
          <td>26.669016</td>
          <td>0.141443</td>
          <td>25.778057</td>
          <td>0.105527</td>
          <td>25.519397</td>
          <td>0.159076</td>
          <td>25.766996</td>
          <td>0.412100</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.330933</td>
          <td>0.330177</td>
          <td>26.507376</td>
          <td>0.139374</td>
          <td>26.223177</td>
          <td>0.095964</td>
          <td>25.657219</td>
          <td>0.094926</td>
          <td>25.174106</td>
          <td>0.118094</td>
          <td>24.835689</td>
          <td>0.194277</td>
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
          <td>27.417712</td>
          <td>0.801442</td>
          <td>26.671836</td>
          <td>0.184286</td>
          <td>25.938502</td>
          <td>0.087845</td>
          <td>25.362225</td>
          <td>0.086701</td>
          <td>24.858327</td>
          <td>0.105259</td>
          <td>24.950664</td>
          <td>0.250392</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.407606</td>
          <td>2.233558</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.405037</td>
          <td>0.304954</td>
          <td>26.860340</td>
          <td>0.309067</td>
          <td>28.696712</td>
          <td>1.628606</td>
          <td>26.014048</td>
          <td>0.569953</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>30.191657</td>
          <td>2.961670</td>
          <td>25.832319</td>
          <td>0.091061</td>
          <td>24.856544</td>
          <td>0.034485</td>
          <td>23.850594</td>
          <td>0.023408</td>
          <td>23.166658</td>
          <td>0.024243</td>
          <td>22.833844</td>
          <td>0.040798</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>31.161071</td>
          <td>3.926730</td>
          <td>29.012250</td>
          <td>1.077909</td>
          <td>27.640899</td>
          <td>0.389969</td>
          <td>26.401986</td>
          <td>0.226722</td>
          <td>26.189959</td>
          <td>0.342881</td>
          <td>25.346700</td>
          <td>0.366360</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.212649</td>
          <td>0.334283</td>
          <td>25.708240</td>
          <td>0.080040</td>
          <td>25.389750</td>
          <td>0.054084</td>
          <td>24.875083</td>
          <td>0.056372</td>
          <td>24.402298</td>
          <td>0.070492</td>
          <td>23.768676</td>
          <td>0.091193</td>
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
          <td>26.949497</td>
          <td>0.589952</td>
          <td>26.305784</td>
          <td>0.137331</td>
          <td>26.340200</td>
          <td>0.127407</td>
          <td>26.111312</td>
          <td>0.169644</td>
          <td>25.994380</td>
          <td>0.281162</td>
          <td>25.701980</td>
          <td>0.461509</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.282114</td>
          <td>0.734594</td>
          <td>26.676246</td>
          <td>0.185644</td>
          <td>26.740673</td>
          <td>0.176704</td>
          <td>26.402593</td>
          <td>0.213281</td>
          <td>25.902541</td>
          <td>0.256852</td>
          <td>25.605218</td>
          <td>0.422614</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.083202</td>
          <td>0.644755</td>
          <td>27.218993</td>
          <td>0.292868</td>
          <td>26.556110</td>
          <td>0.152235</td>
          <td>26.411693</td>
          <td>0.216733</td>
          <td>26.087965</td>
          <td>0.300947</td>
          <td>24.877580</td>
          <td>0.238685</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.453314</td>
          <td>1.479220</td>
          <td>27.264222</td>
          <td>0.308425</td>
          <td>26.667669</td>
          <td>0.170512</td>
          <td>25.814382</td>
          <td>0.132886</td>
          <td>26.024634</td>
          <td>0.290901</td>
          <td>25.665476</td>
          <td>0.453146</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.446591</td>
          <td>0.403810</td>
          <td>26.759065</td>
          <td>0.200062</td>
          <td>25.894098</td>
          <td>0.085333</td>
          <td>25.753130</td>
          <td>0.123303</td>
          <td>25.107977</td>
          <td>0.132081</td>
          <td>25.176680</td>
          <td>0.303737</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.864733</td>
          <td>0.189060</td>
          <td>26.097174</td>
          <td>0.085912</td>
          <td>25.395640</td>
          <td>0.075397</td>
          <td>24.980834</td>
          <td>0.099769</td>
          <td>24.837736</td>
          <td>0.194637</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.589250</td>
          <td>0.714133</td>
          <td>28.171029</td>
          <td>0.479086</td>
          <td>27.623917</td>
          <td>0.481422</td>
          <td>26.796772</td>
          <td>0.448585</td>
          <td>25.878987</td>
          <td>0.449094</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.241610</td>
          <td>0.324290</td>
          <td>26.205918</td>
          <td>0.115254</td>
          <td>24.793467</td>
          <td>0.029440</td>
          <td>23.866194</td>
          <td>0.021372</td>
          <td>23.147294</td>
          <td>0.021561</td>
          <td>22.834857</td>
          <td>0.036714</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>30.411697</td>
          <td>3.202992</td>
          <td>28.135205</td>
          <td>0.610706</td>
          <td>27.824213</td>
          <td>0.447205</td>
          <td>26.991214</td>
          <td>0.363621</td>
          <td>25.867194</td>
          <td>0.263703</td>
          <td>24.922492</td>
          <td>0.260025</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.431277</td>
          <td>0.357660</td>
          <td>25.715012</td>
          <td>0.069789</td>
          <td>25.552303</td>
          <td>0.053105</td>
          <td>24.871734</td>
          <td>0.047443</td>
          <td>24.333857</td>
          <td>0.056418</td>
          <td>23.779597</td>
          <td>0.077955</td>
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
          <td>26.739944</td>
          <td>0.475027</td>
          <td>26.354595</td>
          <td>0.130729</td>
          <td>26.005264</td>
          <td>0.085734</td>
          <td>26.067830</td>
          <td>0.147170</td>
          <td>26.102183</td>
          <td>0.278843</td>
          <td>26.178467</td>
          <td>0.597850</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.349893</td>
          <td>0.707336</td>
          <td>27.170517</td>
          <td>0.247239</td>
          <td>26.789567</td>
          <td>0.159405</td>
          <td>26.497554</td>
          <td>0.199136</td>
          <td>26.314509</td>
          <td>0.312539</td>
          <td>25.039918</td>
          <td>0.234172</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.456689</td>
          <td>0.320004</td>
          <td>27.148044</td>
          <td>0.222538</td>
          <td>26.433414</td>
          <td>0.194910</td>
          <td>25.639141</td>
          <td>0.184655</td>
          <td>25.903765</td>
          <td>0.477330</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.438022</td>
          <td>1.418407</td>
          <td>27.105109</td>
          <td>0.254050</td>
          <td>26.478634</td>
          <td>0.134325</td>
          <td>26.050697</td>
          <td>0.150423</td>
          <td>25.937003</td>
          <td>0.251931</td>
          <td>25.519570</td>
          <td>0.377845</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.852693</td>
          <td>0.987532</td>
          <td>26.473290</td>
          <td>0.139921</td>
          <td>26.054607</td>
          <td>0.086035</td>
          <td>25.596780</td>
          <td>0.093768</td>
          <td>25.229312</td>
          <td>0.128741</td>
          <td>24.964807</td>
          <td>0.224918</td>
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
