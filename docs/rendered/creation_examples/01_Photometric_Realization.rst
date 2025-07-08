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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f1b99b15060>



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
    0      23.994413  0.002547  0.002295  
    1      25.391064  0.073764  0.042009  
    2      24.304707  0.042497  0.024240  
    3      25.291103  0.020630  0.012110  
    4      25.096743  0.203164  0.151934  
    ...          ...       ...       ...  
    99995  24.737946  0.195906  0.171913  
    99996  24.224169  0.014252  0.009252  
    99997  25.613836  0.000639  0.000611  
    99998  25.274899  0.022662  0.016490  
    99999  25.699642  0.047547  0.039321  
    
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
          <td>27.383159</td>
          <td>0.717051</td>
          <td>26.507051</td>
          <td>0.139335</td>
          <td>26.026240</td>
          <td>0.080694</td>
          <td>25.239717</td>
          <td>0.065668</td>
          <td>24.850424</td>
          <td>0.088964</td>
          <td>24.097050</td>
          <td>0.102904</td>
          <td>0.002547</td>
          <td>0.002295</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.815213</td>
          <td>0.947167</td>
          <td>27.645078</td>
          <td>0.357397</td>
          <td>26.531081</td>
          <td>0.125545</td>
          <td>26.343950</td>
          <td>0.172003</td>
          <td>25.940305</td>
          <td>0.226850</td>
          <td>24.892005</td>
          <td>0.203691</td>
          <td>0.073764</td>
          <td>0.042009</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.955721</td>
          <td>0.905607</td>
          <td>28.982558</td>
          <td>0.841055</td>
          <td>26.052063</td>
          <td>0.133916</td>
          <td>24.946081</td>
          <td>0.096763</td>
          <td>24.733127</td>
          <td>0.178149</td>
          <td>0.042497</td>
          <td>0.024240</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.674485</td>
          <td>0.867529</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.285142</td>
          <td>0.238081</td>
          <td>26.342594</td>
          <td>0.171804</td>
          <td>25.867236</td>
          <td>0.213460</td>
          <td>25.270155</td>
          <td>0.278339</td>
          <td>0.020630</td>
          <td>0.012110</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.427364</td>
          <td>0.356252</td>
          <td>26.096791</td>
          <td>0.097542</td>
          <td>25.891138</td>
          <td>0.071614</td>
          <td>25.718307</td>
          <td>0.100150</td>
          <td>25.251044</td>
          <td>0.126253</td>
          <td>25.006669</td>
          <td>0.224155</td>
          <td>0.203164</td>
          <td>0.151934</td>
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
          <td>27.247620</td>
          <td>0.653691</td>
          <td>26.413493</td>
          <td>0.128523</td>
          <td>25.409613</td>
          <td>0.046718</td>
          <td>25.077543</td>
          <td>0.056870</td>
          <td>25.003718</td>
          <td>0.101776</td>
          <td>24.567721</td>
          <td>0.154723</td>
          <td>0.195906</td>
          <td>0.171913</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.973660</td>
          <td>0.207156</td>
          <td>26.117261</td>
          <td>0.087434</td>
          <td>25.130091</td>
          <td>0.059585</td>
          <td>24.941831</td>
          <td>0.096403</td>
          <td>24.331167</td>
          <td>0.126185</td>
          <td>0.014252</td>
          <td>0.009252</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.624454</td>
          <td>0.414979</td>
          <td>26.608905</td>
          <td>0.152079</td>
          <td>26.253133</td>
          <td>0.098519</td>
          <td>26.137254</td>
          <td>0.144125</td>
          <td>25.747684</td>
          <td>0.193093</td>
          <td>25.223411</td>
          <td>0.267953</td>
          <td>0.000639</td>
          <td>0.000611</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.288204</td>
          <td>0.319157</td>
          <td>26.362578</td>
          <td>0.122977</td>
          <td>26.020637</td>
          <td>0.080296</td>
          <td>26.025673</td>
          <td>0.130894</td>
          <td>25.953195</td>
          <td>0.229289</td>
          <td>25.233422</td>
          <td>0.270148</td>
          <td>0.022662</td>
          <td>0.016490</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.892762</td>
          <td>0.507464</td>
          <td>26.762680</td>
          <td>0.173397</td>
          <td>26.601143</td>
          <td>0.133397</td>
          <td>26.444126</td>
          <td>0.187243</td>
          <td>26.177845</td>
          <td>0.275738</td>
          <td>25.843788</td>
          <td>0.436932</td>
          <td>0.047547</td>
          <td>0.039321</td>
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
          <td>27.411241</td>
          <td>0.798066</td>
          <td>26.873299</td>
          <td>0.218219</td>
          <td>25.963980</td>
          <td>0.089835</td>
          <td>25.195405</td>
          <td>0.074838</td>
          <td>24.765615</td>
          <td>0.097052</td>
          <td>24.046956</td>
          <td>0.116283</td>
          <td>0.002547</td>
          <td>0.002295</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.287533</td>
          <td>0.654448</td>
          <td>26.219045</td>
          <td>0.113668</td>
          <td>26.132511</td>
          <td>0.171191</td>
          <td>26.196058</td>
          <td>0.327878</td>
          <td>25.042114</td>
          <td>0.272958</td>
          <td>0.073764</td>
          <td>0.042009</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.689770</td>
          <td>0.483320</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.101996</td>
          <td>0.522721</td>
          <td>25.841305</td>
          <td>0.132275</td>
          <td>25.175975</td>
          <td>0.139242</td>
          <td>24.110960</td>
          <td>0.123439</td>
          <td>0.042497</td>
          <td>0.024240</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.642404</td>
          <td>0.465632</td>
          <td>28.315735</td>
          <td>0.662079</td>
          <td>27.252829</td>
          <td>0.269824</td>
          <td>26.454254</td>
          <td>0.221960</td>
          <td>25.417251</td>
          <td>0.170689</td>
          <td>25.680352</td>
          <td>0.446125</td>
          <td>0.020630</td>
          <td>0.012110</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.955771</td>
          <td>0.621932</td>
          <td>25.921350</td>
          <td>0.105472</td>
          <td>26.020104</td>
          <td>0.104082</td>
          <td>25.611434</td>
          <td>0.119270</td>
          <td>25.435488</td>
          <td>0.190346</td>
          <td>25.319743</td>
          <td>0.368948</td>
          <td>0.203164</td>
          <td>0.151934</td>
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
          <td>26.795626</td>
          <td>0.556989</td>
          <td>26.199496</td>
          <td>0.134926</td>
          <td>25.436622</td>
          <td>0.062586</td>
          <td>25.061593</td>
          <td>0.074048</td>
          <td>24.862409</td>
          <td>0.117064</td>
          <td>24.592532</td>
          <td>0.205621</td>
          <td>0.195906</td>
          <td>0.171913</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.940881</td>
          <td>0.578954</td>
          <td>26.497537</td>
          <td>0.158983</td>
          <td>26.203679</td>
          <td>0.110872</td>
          <td>25.161773</td>
          <td>0.072682</td>
          <td>24.902901</td>
          <td>0.109487</td>
          <td>24.155989</td>
          <td>0.127888</td>
          <td>0.014252</td>
          <td>0.009252</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.838484</td>
          <td>0.537678</td>
          <td>26.619540</td>
          <td>0.176302</td>
          <td>26.275276</td>
          <td>0.117948</td>
          <td>25.992865</td>
          <td>0.150102</td>
          <td>25.976845</td>
          <td>0.271856</td>
          <td>25.125868</td>
          <td>0.288808</td>
          <td>0.000639</td>
          <td>0.000611</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.861464</td>
          <td>0.252073</td>
          <td>26.306846</td>
          <td>0.135075</td>
          <td>26.156470</td>
          <td>0.106485</td>
          <td>25.804571</td>
          <td>0.127781</td>
          <td>25.628753</td>
          <td>0.204141</td>
          <td>25.891132</td>
          <td>0.521931</td>
          <td>0.022662</td>
          <td>0.016490</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.953793</td>
          <td>0.586447</td>
          <td>27.182272</td>
          <td>0.282806</td>
          <td>26.990189</td>
          <td>0.218427</td>
          <td>26.277697</td>
          <td>0.192493</td>
          <td>25.895438</td>
          <td>0.255894</td>
          <td>26.341816</td>
          <td>0.719404</td>
          <td>0.047547</td>
          <td>0.039321</td>
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
          <td>26.489518</td>
          <td>0.373986</td>
          <td>26.504131</td>
          <td>0.138995</td>
          <td>26.074791</td>
          <td>0.084230</td>
          <td>25.161700</td>
          <td>0.061284</td>
          <td>24.720748</td>
          <td>0.079364</td>
          <td>24.041897</td>
          <td>0.098059</td>
          <td>0.002547</td>
          <td>0.002295</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.145614</td>
          <td>0.624763</td>
          <td>27.242806</td>
          <td>0.268727</td>
          <td>26.963197</td>
          <td>0.190100</td>
          <td>26.302584</td>
          <td>0.173999</td>
          <td>25.565910</td>
          <td>0.173070</td>
          <td>25.554403</td>
          <td>0.364601</td>
          <td>0.073764</td>
          <td>0.042009</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.080932</td>
          <td>0.987563</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.093106</td>
          <td>0.141041</td>
          <td>25.049884</td>
          <td>0.107662</td>
          <td>24.247979</td>
          <td>0.119330</td>
          <td>0.042497</td>
          <td>0.024240</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.388152</td>
          <td>0.346277</td>
          <td>29.430501</td>
          <td>1.201528</td>
          <td>27.468957</td>
          <td>0.277769</td>
          <td>26.214555</td>
          <td>0.154628</td>
          <td>25.400648</td>
          <td>0.144215</td>
          <td>25.520699</td>
          <td>0.341444</td>
          <td>0.020630</td>
          <td>0.012110</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.011602</td>
          <td>0.666157</td>
          <td>26.321175</td>
          <td>0.155496</td>
          <td>25.918935</td>
          <td>0.099791</td>
          <td>25.612266</td>
          <td>0.125123</td>
          <td>25.333517</td>
          <td>0.182627</td>
          <td>24.552233</td>
          <td>0.206897</td>
          <td>0.203164</td>
          <td>0.151934</td>
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
          <td>27.875440</td>
          <td>1.156102</td>
          <td>26.504394</td>
          <td>0.184275</td>
          <td>25.511386</td>
          <td>0.070833</td>
          <td>25.137967</td>
          <td>0.084014</td>
          <td>24.949888</td>
          <td>0.133599</td>
          <td>25.914748</td>
          <td>0.606452</td>
          <td>0.195906</td>
          <td>0.171913</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.787036</td>
          <td>0.177308</td>
          <td>26.102596</td>
          <td>0.086482</td>
          <td>25.295178</td>
          <td>0.069118</td>
          <td>24.848327</td>
          <td>0.088974</td>
          <td>24.131509</td>
          <td>0.106267</td>
          <td>0.014252</td>
          <td>0.009252</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.337300</td>
          <td>0.695144</td>
          <td>26.529752</td>
          <td>0.142086</td>
          <td>26.142461</td>
          <td>0.089395</td>
          <td>26.508864</td>
          <td>0.197743</td>
          <td>25.698957</td>
          <td>0.185315</td>
          <td>25.521818</td>
          <td>0.340515</td>
          <td>0.000639</td>
          <td>0.000611</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.097148</td>
          <td>0.274625</td>
          <td>26.154314</td>
          <td>0.103049</td>
          <td>26.127860</td>
          <td>0.088724</td>
          <td>25.698393</td>
          <td>0.098966</td>
          <td>25.661341</td>
          <td>0.180427</td>
          <td>26.089050</td>
          <td>0.526839</td>
          <td>0.022662</td>
          <td>0.016490</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.735814</td>
          <td>0.458385</td>
          <td>26.737988</td>
          <td>0.173433</td>
          <td>26.597547</td>
          <td>0.136317</td>
          <td>26.211863</td>
          <td>0.157668</td>
          <td>26.003433</td>
          <td>0.244729</td>
          <td>25.761172</td>
          <td>0.419705</td>
          <td>0.047547</td>
          <td>0.039321</td>
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
