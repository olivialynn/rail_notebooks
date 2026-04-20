Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``00_Quick_Start_in_Creation.ipynb``

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`Photometric_Realization.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/creation_examples/Photometric_Realization.ipynb>`__
in the ``interactive_examples/creation_examples/`` folder for a version
of this notebook in interactive mode.

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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fbd009ea9e0>



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
    0      23.994413  0.000130  0.000117  
    1      25.391064  0.013036  0.008422  
    2      24.304707  0.064522  0.060899  
    3      25.291103  0.145166  0.104599  
    4      25.096743  0.097783  0.090336  
    ...          ...       ...       ...  
    99995  24.737946  0.232038  0.136058  
    99996  24.224169  0.026279  0.015279  
    99997  25.613836  0.027437  0.020286  
    99998  25.274899  0.084843  0.044437  
    99999  25.699642  0.064724  0.057962  
    
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

    Inserting handle into data store.  output_truth: None, error_model
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
          <td>27.296148</td>
          <td>0.675892</td>
          <td>26.890018</td>
          <td>0.193110</td>
          <td>25.982176</td>
          <td>0.077615</td>
          <td>25.263424</td>
          <td>0.067062</td>
          <td>24.644698</td>
          <td>0.074202</td>
          <td>23.995500</td>
          <td>0.094140</td>
          <td>0.000130</td>
          <td>0.000117</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.198695</td>
          <td>0.631852</td>
          <td>27.996710</td>
          <td>0.467938</td>
          <td>26.712885</td>
          <td>0.146884</td>
          <td>26.463868</td>
          <td>0.190389</td>
          <td>26.133319</td>
          <td>0.265919</td>
          <td>25.123488</td>
          <td>0.246893</td>
          <td>0.013036</td>
          <td>0.008422</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.077513</td>
          <td>0.446289</td>
          <td>26.006058</td>
          <td>0.128690</td>
          <td>25.041636</td>
          <td>0.105209</td>
          <td>24.448142</td>
          <td>0.139614</td>
          <td>0.064522</td>
          <td>0.060899</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.865319</td>
          <td>0.497311</td>
          <td>27.834545</td>
          <td>0.413908</td>
          <td>27.078795</td>
          <td>0.200477</td>
          <td>26.241766</td>
          <td>0.157646</td>
          <td>25.508137</td>
          <td>0.157551</td>
          <td>25.539973</td>
          <td>0.345427</td>
          <td>0.145166</td>
          <td>0.104599</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.523625</td>
          <td>0.384000</td>
          <td>26.113868</td>
          <td>0.099012</td>
          <td>25.912734</td>
          <td>0.072995</td>
          <td>25.643195</td>
          <td>0.093764</td>
          <td>25.599049</td>
          <td>0.170259</td>
          <td>25.195718</td>
          <td>0.261962</td>
          <td>0.097783</td>
          <td>0.090336</td>
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
          <td>26.968391</td>
          <td>0.536289</td>
          <td>26.407462</td>
          <td>0.127853</td>
          <td>25.382589</td>
          <td>0.045611</td>
          <td>25.139209</td>
          <td>0.060069</td>
          <td>24.810727</td>
          <td>0.085909</td>
          <td>24.640681</td>
          <td>0.164679</td>
          <td>0.232038</td>
          <td>0.136058</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.559701</td>
          <td>0.805912</td>
          <td>27.005730</td>
          <td>0.212784</td>
          <td>25.919843</td>
          <td>0.073455</td>
          <td>25.092738</td>
          <td>0.057642</td>
          <td>25.024897</td>
          <td>0.103680</td>
          <td>24.411572</td>
          <td>0.135277</td>
          <td>0.026279</td>
          <td>0.015279</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.680833</td>
          <td>0.433180</td>
          <td>26.859563</td>
          <td>0.188216</td>
          <td>26.488405</td>
          <td>0.120980</td>
          <td>26.184062</td>
          <td>0.150041</td>
          <td>25.784095</td>
          <td>0.199100</td>
          <td>25.627555</td>
          <td>0.369986</td>
          <td>0.027437</td>
          <td>0.020286</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.366041</td>
          <td>0.339476</td>
          <td>26.135582</td>
          <td>0.100911</td>
          <td>26.291497</td>
          <td>0.101886</td>
          <td>25.821691</td>
          <td>0.109627</td>
          <td>25.663191</td>
          <td>0.179788</td>
          <td>25.153334</td>
          <td>0.253024</td>
          <td>0.084843</td>
          <td>0.044437</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.341737</td>
          <td>0.697240</td>
          <td>27.008496</td>
          <td>0.213276</td>
          <td>26.421057</td>
          <td>0.114094</td>
          <td>26.389129</td>
          <td>0.178729</td>
          <td>25.962657</td>
          <td>0.231094</td>
          <td>26.381128</td>
          <td>0.645735</td>
          <td>0.064724</td>
          <td>0.057962</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_truth: None, error_model_gaap


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
          <td>inf</td>
          <td>inf</td>
          <td>26.907220</td>
          <td>0.224458</td>
          <td>25.952552</td>
          <td>0.088935</td>
          <td>25.241641</td>
          <td>0.077955</td>
          <td>24.771957</td>
          <td>0.097591</td>
          <td>24.012748</td>
          <td>0.112868</td>
          <td>0.000130</td>
          <td>0.000117</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.980383</td>
          <td>1.130889</td>
          <td>27.892044</td>
          <td>0.488682</td>
          <td>26.685562</td>
          <td>0.168000</td>
          <td>26.321980</td>
          <td>0.198609</td>
          <td>26.073295</td>
          <td>0.294058</td>
          <td>27.792373</td>
          <td>1.645290</td>
          <td>0.013036</td>
          <td>0.008422</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.830074</td>
          <td>1.758423</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.298930</td>
          <td>1.148396</td>
          <td>26.114954</td>
          <td>0.168847</td>
          <td>25.170593</td>
          <td>0.139850</td>
          <td>24.111643</td>
          <td>0.124667</td>
          <td>0.064522</td>
          <td>0.060899</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.236360</td>
          <td>0.306654</td>
          <td>27.099262</td>
          <td>0.249404</td>
          <td>26.315903</td>
          <td>0.207796</td>
          <td>25.399434</td>
          <td>0.176523</td>
          <td>25.354392</td>
          <td>0.363316</td>
          <td>0.145166</td>
          <td>0.104599</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.882011</td>
          <td>0.261503</td>
          <td>26.150932</td>
          <td>0.121009</td>
          <td>26.099945</td>
          <td>0.104213</td>
          <td>25.757739</td>
          <td>0.126250</td>
          <td>25.187008</td>
          <td>0.144089</td>
          <td>25.092097</td>
          <td>0.288913</td>
          <td>0.097783</td>
          <td>0.090336</td>
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
          <td>26.632901</td>
          <td>0.496067</td>
          <td>26.441160</td>
          <td>0.166565</td>
          <td>25.308501</td>
          <td>0.056081</td>
          <td>25.024355</td>
          <td>0.071927</td>
          <td>24.740411</td>
          <td>0.105648</td>
          <td>24.841686</td>
          <td>0.253715</td>
          <td>0.232038</td>
          <td>0.136058</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.721753</td>
          <td>0.494117</td>
          <td>26.610180</td>
          <td>0.175147</td>
          <td>26.109620</td>
          <td>0.102235</td>
          <td>25.100902</td>
          <td>0.068950</td>
          <td>24.842680</td>
          <td>0.103988</td>
          <td>24.405988</td>
          <td>0.158762</td>
          <td>0.026279</td>
          <td>0.015279</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.456231</td>
          <td>0.822599</td>
          <td>26.595848</td>
          <td>0.173091</td>
          <td>26.594199</td>
          <td>0.155630</td>
          <td>26.206814</td>
          <td>0.180501</td>
          <td>26.053977</td>
          <td>0.289927</td>
          <td>25.098449</td>
          <td>0.283002</td>
          <td>0.027437</td>
          <td>0.020286</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.265935</td>
          <td>0.352205</td>
          <td>26.358770</td>
          <td>0.143013</td>
          <td>26.259979</td>
          <td>0.118167</td>
          <td>25.902492</td>
          <td>0.141051</td>
          <td>25.716418</td>
          <td>0.222577</td>
          <td>24.918971</td>
          <td>0.247560</td>
          <td>0.084843</td>
          <td>0.044437</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.191579</td>
          <td>0.694529</td>
          <td>26.790712</td>
          <td>0.205905</td>
          <td>26.881602</td>
          <td>0.200654</td>
          <td>27.112119</td>
          <td>0.381354</td>
          <td>26.426512</td>
          <td>0.392973</td>
          <td>26.125029</td>
          <td>0.622969</td>
          <td>0.064724</td>
          <td>0.057962</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_truth: None, error_model_auto


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
          <td>27.664168</td>
          <td>0.861868</td>
          <td>26.772415</td>
          <td>0.174836</td>
          <td>25.911797</td>
          <td>0.072935</td>
          <td>25.081514</td>
          <td>0.057071</td>
          <td>24.806416</td>
          <td>0.085584</td>
          <td>24.020601</td>
          <td>0.096237</td>
          <td>0.000130</td>
          <td>0.000117</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.390409</td>
          <td>0.292226</td>
          <td>26.733831</td>
          <td>0.149790</td>
          <td>26.466357</td>
          <td>0.191104</td>
          <td>25.779916</td>
          <td>0.198712</td>
          <td>25.093635</td>
          <td>0.241278</td>
          <td>0.013036</td>
          <td>0.008422</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.999384</td>
          <td>0.439439</td>
          <td>25.998357</td>
          <td>0.134703</td>
          <td>25.145964</td>
          <td>0.121165</td>
          <td>24.239875</td>
          <td>0.122796</td>
          <td>0.064522</td>
          <td>0.060899</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.485760</td>
          <td>0.750481</td>
          <td>27.332101</td>
          <td>0.291604</td>
          <td>26.193077</td>
          <td>0.180811</td>
          <td>25.387949</td>
          <td>0.168834</td>
          <td>24.669744</td>
          <td>0.201257</td>
          <td>0.145166</td>
          <td>0.104599</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.121951</td>
          <td>0.299667</td>
          <td>25.995464</td>
          <td>0.098083</td>
          <td>25.998640</td>
          <td>0.087720</td>
          <td>25.576088</td>
          <td>0.098892</td>
          <td>25.474497</td>
          <td>0.169900</td>
          <td>24.761256</td>
          <td>0.202965</td>
          <td>0.097783</td>
          <td>0.090336</td>
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
          <td>27.040353</td>
          <td>0.687546</td>
          <td>26.198882</td>
          <td>0.142267</td>
          <td>25.582807</td>
          <td>0.075514</td>
          <td>24.927229</td>
          <td>0.069794</td>
          <td>24.973609</td>
          <td>0.136483</td>
          <td>24.878429</td>
          <td>0.275159</td>
          <td>0.232038</td>
          <td>0.136058</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.949775</td>
          <td>0.530975</td>
          <td>26.429227</td>
          <td>0.130979</td>
          <td>26.097716</td>
          <td>0.086481</td>
          <td>25.202936</td>
          <td>0.063984</td>
          <td>24.717322</td>
          <td>0.079615</td>
          <td>24.338921</td>
          <td>0.127850</td>
          <td>0.026279</td>
          <td>0.015279</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.244034</td>
          <td>1.220857</td>
          <td>26.749659</td>
          <td>0.172622</td>
          <td>26.228428</td>
          <td>0.097166</td>
          <td>26.542793</td>
          <td>0.205062</td>
          <td>26.344903</td>
          <td>0.317721</td>
          <td>25.571970</td>
          <td>0.356821</td>
          <td>0.027437</td>
          <td>0.020286</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.198547</td>
          <td>0.308333</td>
          <td>26.190055</td>
          <td>0.111270</td>
          <td>26.180233</td>
          <td>0.097889</td>
          <td>25.773587</td>
          <td>0.111607</td>
          <td>25.752499</td>
          <td>0.204836</td>
          <td>24.894974</td>
          <td>0.216130</td>
          <td>0.084843</td>
          <td>0.044437</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.434564</td>
          <td>0.761625</td>
          <td>26.846295</td>
          <td>0.193870</td>
          <td>26.525257</td>
          <td>0.131061</td>
          <td>26.282463</td>
          <td>0.171506</td>
          <td>25.925859</td>
          <td>0.234673</td>
          <td>25.391136</td>
          <td>0.321303</td>
          <td>0.064724</td>
          <td>0.057962</td>
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




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_22_0.png


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




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_23_0.png


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
