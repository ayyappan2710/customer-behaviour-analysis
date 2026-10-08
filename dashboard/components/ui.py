import streamlit as st


def insight_card(title: str, insight: str, action: str):
    st.markdown(f"""
<div class="gc" style="border-left:3px solid #6C63FF;padding:20px 24px;margin-bottom:12px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
    <span style="color:#6C63FF;font-size:1.1rem;">✦</span>
    <strong style="font-size:1rem;color:#EAEDF2;">{title}</strong>
  </div>
  <p style="margin:0 0 12px;color:#EAEDF2;font-size:0.95rem;line-height:1.55;">{insight}</p>
  <div style="background:rgba(108,99,255,0.1);border-radius:8px;padding:10px 14px;">
    <span style="font-size:0.85rem;font-weight:600;color:#3B82F6;">→ {action}</span>
  </div>
</div>""", unsafe_allow_html=True)


def rec_card(rank: int, category: str, score: float, reason: str):
    pct = int(score)
    if pct >= 80:
        clr = "#22C55E"
    elif pct >= 50:
        clr = "#6C63FF"
    else:
        clr = "#F59E0B"

    st.markdown(f"""
<div class="gc" style="margin-bottom:14px;padding:20px 24px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
    <div style="display:flex;align-items:center;gap:12px;">
      <span style="background:rgba(108,99,255,0.15);color:#8B92A5;font-size:0.78rem;font-weight:700;
                   padding:3px 9px;border-radius:6px;border:1px solid rgba(108,99,255,0.2);">#{rank}</span>
      <span style="font-size:1.05rem;font-weight:600;color:#EAEDF2;">{category}</span>
    </div>
    <span style="font-size:1.05rem;font-weight:700;color:{clr};">{pct}% Match</span>
  </div>
  <div class="rec-bar-track">
    <div class="rec-bar-fill" style="width:{pct}%;background:{clr};"></div>
  </div>
  <p style="margin:10px 0 0;font-size:0.88rem;color:#8B92A5;line-height:1.45;">
    <span style="color:{clr};margin-right:5px;">●</span>{reason}
  </p>
</div>""", unsafe_allow_html=True)


def customer_profile_card(cust_id, segment, monetary, frequency, recency, avg_order, last_purchase):
    seg_color = {
        "vip": "#22C55E", "high": "#22C55E",
        "regular": "#6C63FF",
        "low": "#F59E0B", "at risk": "#EF4444",
    }
    seg_key = str(segment).lower()
    color = next((v for k, v in seg_color.items() if k in seg_key), "#8B92A5")

    st.markdown(f"""
<div class="gc" style="padding:24px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
    <div>
      <div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:#8B92A5;">Customer ID</div>
      <div style="font-size:1.3rem;font-weight:700;color:#EAEDF2;">{cust_id}</div>
    </div>
    <span style="background:{color}22;color:{color};font-size:0.82rem;font-weight:600;
                 padding:5px 14px;border-radius:20px;border:1px solid {color}44;">
      {segment}
    </span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;">
    <div>
      <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:.07em;color:#8B92A5;margin-bottom:4px;">Total Spend</div>
      <div style="font-size:1.15rem;font-weight:600;color:#EAEDF2;">${monetary:,.0f}</div>
    </div>
    <div>
      <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:.07em;color:#8B92A5;margin-bottom:4px;">Orders</div>
      <div style="font-size:1.15rem;font-weight:600;color:#EAEDF2;">{int(frequency)}</div>
    </div>
    <div>
      <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:.07em;color:#8B92A5;margin-bottom:4px;">Avg Order</div>
      <div style="font-size:1.15rem;font-weight:600;color:#EAEDF2;">${avg_order:,.0f}</div>
    </div>
    <div>
      <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:.07em;color:#8B92A5;margin-bottom:4px;">Recency (days)</div>
      <div style="font-size:1.15rem;font-weight:600;color:#EAEDF2;">{int(recency)}</div>
    </div>
    <div>
      <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:.07em;color:#8B92A5;margin-bottom:4px;">Last Purchase</div>
      <div style="font-size:1.15rem;font-weight:600;color:#EAEDF2;">{str(last_purchase)[:10]}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


def probability_display(prob: float, label: str):
    clr = "#22C55E" if prob >= 60 else ("#EF4444" if prob < 40 else "#F59E0B")
    pct_fill = int(prob)
    st.markdown(f"""
<div class="gc" style="padding:32px 24px;text-align:center;">
  <div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;
              color:#8B92A5;margin-bottom:16px;">High Activity Probability</div>
  <div style="font-size:4.5rem;font-weight:700;line-height:1;color:{clr};">{prob}%</div>
  <div style="background:rgba(255,255,255,0.07);border-radius:8px;height:8px;margin:20px auto;max-width:240px;">
    <div style="width:{pct_fill}%;height:8px;border-radius:8px;background:{clr};
                box-shadow:0 0 12px {clr};transition:width 0.6s ease;"></div>
  </div>
  <div style="font-size:1.05rem;font-weight:600;color:{clr};">{label}</div>
</div>""", unsafe_allow_html=True)


def html_table(df, title=None):
    html = '<div class="gc" style="padding:0; overflow:hidden; margin-bottom: 24px;">'
    if title:
        html += f'<div style="padding:16px 20px; font-weight:600; border-bottom:1px solid rgba(255,255,255,0.08); color:#EAEDF2;">{title}</div>'
    
    html += '<div style="overflow-x:auto;"><table style="width:100%; border-collapse:collapse; text-align:left; font-size:0.9rem;">'
    
    # Headers
    html += '<thead><tr style="background:rgba(255,255,255,0.03);">'
    for col in df.columns:
        html += f'<th style="padding:12px 20px; color:#8B92A5; font-weight:600; text-transform:uppercase; font-size:0.75rem; letter-spacing:0.05em; border-bottom:1px solid rgba(255,255,255,0.08);">{col}</th>'
    html += '</tr></thead>'
    
    # Body
    html += '<tbody>'
    for i, row in df.iterrows():
        bg = 'rgba(255,255,255,0.015)' if i % 2 == 0 else 'transparent'
        html += f'<tr style="background:{bg};">'
        for val in row:
            html += f'<td style="padding:14px 20px; color:#EAEDF2; border-bottom:1px solid rgba(255,255,255,0.04);">{val}</td>'
        html += '</tr>'
    html += '</tbody></table></div></div>'
    st.markdown(html, unsafe_allow_html=True)
