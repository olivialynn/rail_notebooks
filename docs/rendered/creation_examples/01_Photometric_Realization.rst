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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f623ffbc850>



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
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>1.398944</td>
          <td>27.129906</td>
          <td>0.602064</td>
          <td>27.126971</td>
          <td>0.235335</td>
          <td>26.050067</td>
          <td>0.082408</td>
          <td>25.155776</td>
          <td>0.060958</td>
          <td>24.616212</td>
          <td>0.072356</td>
          <td>24.212335</td>
          <td>0.113803</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.883800</td>
          <td>2.537960</td>
          <td>27.792983</td>
          <td>0.400915</td>
          <td>26.484800</td>
          <td>0.120601</td>
          <td>26.088620</td>
          <td>0.138210</td>
          <td>25.797393</td>
          <td>0.201336</td>
          <td>25.293244</td>
          <td>0.283598</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.367257</td>
          <td>0.612416</td>
          <td>27.981061</td>
          <td>0.414743</td>
          <td>25.897768</td>
          <td>0.117143</td>
          <td>24.945639</td>
          <td>0.096725</td>
          <td>24.270701</td>
          <td>0.119733</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.160952</td>
          <td>1.161418</td>
          <td>28.312669</td>
          <td>0.589229</td>
          <td>27.176681</td>
          <td>0.217588</td>
          <td>26.627033</td>
          <td>0.218303</td>
          <td>25.494218</td>
          <td>0.155686</td>
          <td>25.582114</td>
          <td>0.357066</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.393446</td>
          <td>0.346888</td>
          <td>26.127243</td>
          <td>0.100177</td>
          <td>25.922844</td>
          <td>0.073651</td>
          <td>25.780215</td>
          <td>0.105726</td>
          <td>25.600695</td>
          <td>0.170497</td>
          <td>24.901564</td>
          <td>0.205330</td>
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
          <td>0.389450</td>
          <td>26.474665</td>
          <td>0.369668</td>
          <td>26.314769</td>
          <td>0.117977</td>
          <td>25.464495</td>
          <td>0.049051</td>
          <td>25.056783</td>
          <td>0.055831</td>
          <td>24.971469</td>
          <td>0.098941</td>
          <td>24.757758</td>
          <td>0.181905</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.708815</td>
          <td>0.165632</td>
          <td>26.114996</td>
          <td>0.087260</td>
          <td>25.214535</td>
          <td>0.064219</td>
          <td>24.863861</td>
          <td>0.090022</td>
          <td>24.169389</td>
          <td>0.109620</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.883000</td>
          <td>0.503834</td>
          <td>27.082850</td>
          <td>0.226891</td>
          <td>26.544063</td>
          <td>0.126966</td>
          <td>26.126818</td>
          <td>0.142836</td>
          <td>26.209515</td>
          <td>0.282915</td>
          <td>25.976626</td>
          <td>0.482745</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.237196</td>
          <td>0.306418</td>
          <td>26.020956</td>
          <td>0.091266</td>
          <td>26.145158</td>
          <td>0.089606</td>
          <td>26.023340</td>
          <td>0.130630</td>
          <td>25.520595</td>
          <td>0.159239</td>
          <td>25.303155</td>
          <td>0.285883</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.349985</td>
          <td>1.289188</td>
          <td>26.558189</td>
          <td>0.145603</td>
          <td>26.404009</td>
          <td>0.112411</td>
          <td>26.463139</td>
          <td>0.190272</td>
          <td>26.448753</td>
          <td>0.342586</td>
          <td>25.875068</td>
          <td>0.447391</td>
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
          <td>1.398944</td>
          <td>27.404957</td>
          <td>0.794808</td>
          <td>26.648686</td>
          <td>0.180713</td>
          <td>25.903521</td>
          <td>0.085182</td>
          <td>25.078263</td>
          <td>0.067473</td>
          <td>24.753904</td>
          <td>0.096062</td>
          <td>23.767286</td>
          <td>0.091052</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.312908</td>
          <td>1.357127</td>
          <td>29.244607</td>
          <td>1.183515</td>
          <td>26.398967</td>
          <td>0.131336</td>
          <td>26.374286</td>
          <td>0.207481</td>
          <td>25.645423</td>
          <td>0.206796</td>
          <td>25.548133</td>
          <td>0.403106</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.205511</td>
          <td>0.622688</td>
          <td>28.891586</td>
          <td>0.906282</td>
          <td>25.982003</td>
          <td>0.152102</td>
          <td>24.952285</td>
          <td>0.116803</td>
          <td>24.175073</td>
          <td>0.132923</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.134004</td>
          <td>1.272785</td>
          <td>27.746176</td>
          <td>0.461430</td>
          <td>27.706129</td>
          <td>0.410055</td>
          <td>26.055886</td>
          <td>0.169470</td>
          <td>25.691196</td>
          <td>0.228904</td>
          <td>25.595549</td>
          <td>0.443586</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.782066</td>
          <td>0.516133</td>
          <td>25.951053</td>
          <td>0.099057</td>
          <td>25.754870</td>
          <td>0.074737</td>
          <td>25.497323</td>
          <td>0.097658</td>
          <td>25.318764</td>
          <td>0.156840</td>
          <td>24.989464</td>
          <td>0.258568</td>
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
          <td>0.389450</td>
          <td>26.783434</td>
          <td>0.523474</td>
          <td>26.118416</td>
          <td>0.116772</td>
          <td>25.316013</td>
          <td>0.051732</td>
          <td>25.091474</td>
          <td>0.069780</td>
          <td>24.810145</td>
          <td>0.103048</td>
          <td>24.592507</td>
          <td>0.189644</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.722751</td>
          <td>0.495315</td>
          <td>26.791642</td>
          <td>0.204565</td>
          <td>26.094827</td>
          <td>0.101185</td>
          <td>25.315515</td>
          <td>0.083568</td>
          <td>24.875843</td>
          <td>0.107326</td>
          <td>24.243172</td>
          <td>0.138414</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.786807</td>
          <td>0.522032</td>
          <td>26.470964</td>
          <td>0.157106</td>
          <td>26.570721</td>
          <td>0.154153</td>
          <td>26.221832</td>
          <td>0.184795</td>
          <td>27.107551</td>
          <td>0.648076</td>
          <td>25.185119</td>
          <td>0.306588</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.375956</td>
          <td>0.387961</td>
          <td>26.037035</td>
          <td>0.109812</td>
          <td>26.087629</td>
          <td>0.103319</td>
          <td>25.616869</td>
          <td>0.111947</td>
          <td>25.954637</td>
          <td>0.274867</td>
          <td>25.120973</td>
          <td>0.296323</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.720134</td>
          <td>0.496218</td>
          <td>26.965129</td>
          <td>0.237509</td>
          <td>26.692680</td>
          <td>0.170610</td>
          <td>26.323879</td>
          <td>0.200837</td>
          <td>25.994691</td>
          <td>0.278406</td>
          <td>26.264967</td>
          <td>0.684840</td>
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
          <td>1.398944</td>
          <td>26.768009</td>
          <td>0.462634</td>
          <td>26.595328</td>
          <td>0.150336</td>
          <td>26.098098</td>
          <td>0.085982</td>
          <td>25.162908</td>
          <td>0.061354</td>
          <td>24.649955</td>
          <td>0.074557</td>
          <td>23.845968</td>
          <td>0.082545</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.378572</td>
          <td>0.289289</td>
          <td>26.391979</td>
          <td>0.111343</td>
          <td>26.372218</td>
          <td>0.176352</td>
          <td>25.576052</td>
          <td>0.167110</td>
          <td>25.322893</td>
          <td>0.290743</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.452439</td>
          <td>0.784791</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.439798</td>
          <td>1.171527</td>
          <td>26.090162</td>
          <td>0.150485</td>
          <td>25.200377</td>
          <td>0.130978</td>
          <td>24.273754</td>
          <td>0.130529</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.124271</td>
          <td>1.263894</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.544862</td>
          <td>0.360745</td>
          <td>26.330431</td>
          <td>0.212865</td>
          <td>25.605941</td>
          <td>0.212506</td>
          <td>25.238995</td>
          <td>0.335506</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.863642</td>
          <td>0.497106</td>
          <td>26.135714</td>
          <td>0.101047</td>
          <td>25.809063</td>
          <td>0.066690</td>
          <td>25.666255</td>
          <td>0.095824</td>
          <td>25.148980</td>
          <td>0.115702</td>
          <td>25.634301</td>
          <td>0.372424</td>
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
          <td>0.389450</td>
          <td>29.110997</td>
          <td>1.921292</td>
          <td>26.201991</td>
          <td>0.114523</td>
          <td>25.442451</td>
          <td>0.052095</td>
          <td>25.127812</td>
          <td>0.064629</td>
          <td>24.876340</td>
          <td>0.098461</td>
          <td>24.638367</td>
          <td>0.177859</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.069204</td>
          <td>1.110388</td>
          <td>26.711696</td>
          <td>0.168360</td>
          <td>26.086183</td>
          <td>0.086491</td>
          <td>25.198697</td>
          <td>0.064440</td>
          <td>24.953007</td>
          <td>0.098965</td>
          <td>24.500164</td>
          <td>0.148473</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>29.853013</td>
          <td>2.545519</td>
          <td>26.692348</td>
          <td>0.170230</td>
          <td>26.290638</td>
          <td>0.106894</td>
          <td>26.163466</td>
          <td>0.154973</td>
          <td>26.246887</td>
          <td>0.304932</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.864220</td>
          <td>0.532146</td>
          <td>26.170022</td>
          <td>0.114964</td>
          <td>26.041443</td>
          <td>0.091753</td>
          <td>26.176509</td>
          <td>0.167506</td>
          <td>25.466501</td>
          <td>0.169943</td>
          <td>25.931059</td>
          <td>0.515669</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>29.555321</td>
          <td>2.273050</td>
          <td>26.830196</td>
          <td>0.189689</td>
          <td>26.795077</td>
          <td>0.163688</td>
          <td>26.180828</td>
          <td>0.155700</td>
          <td>25.933128</td>
          <td>0.233923</td>
          <td>25.795628</td>
          <td>0.436194</td>
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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


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
